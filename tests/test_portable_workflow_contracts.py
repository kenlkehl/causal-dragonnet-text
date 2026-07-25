from __future__ import annotations

import json
import hashlib
import os
import shutil
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oci.extraction.complete_paged import CompletePagingGeometry
import oci.inference.portable_artifacts as portable_artifacts_module
from oci.inference.physical_fit_deduplication import (
    build_logical_binding_records,
    derive_logical_context_plan,
    group_equivalent_contexts,
)
from oci.inference.legacy_checkpoint_migration import (
    LegacyEmbeddingCacheMigrationExpectation,
    LegacyPreparedMigrationExpectation,
    classify_legacy_workflow,
    migrate_legacy_terminal_phase_reference,
    plan_legacy_v4_preflight_migration,
)
import oci.inference.legacy_checkpoint_migration as legacy_migration_module
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    adopt_checkpoint,
    materialize_portable_phase,
    publish_portable_artifact,
    publish_portable_reference_artifact,
    relocate_portable_artifact,
    validate_portable_artifact,
)
from oci.inference.portable_resource_scheduler import (
    GIB,
    GPUResource,
    ResourceInventory,
    plan_resources,
)
from oci.inference.portable_workflow_spec import (
    BINARY_PROBABILITY_DIFFERENCE,
    DeploymentProfile,
    EVIDENCE_FAMILIES,
    FoldReviewSpec,
    HierarchyWireBudgetSpec,
    LosslessTextWindowSpec,
    PostExtractionCausalReviewSpec,
    PORTABLE_SPEC_VERSION,
    RESOURCE_PERFORMANCE_SAFETY_VERSION,
    ResourcePerformanceSafetyPolicy,
    RUN_CONTROL_VERSION,
    RunControl,
    SentenceEmbeddingEncoderSpec,
    ScientificWorkflowSpec,
    Stage1ExecutionProfile,
    Stage2PromptProtocolSpec,
    StrictCausalForestOperationalSpec,
    StrictCausalForestSpec,
    TextPreprocessingSpec,
    WorkflowColumns,
    compile_strict_causal_forest_runtime,
    identity_sha256,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    Stage2GenerationParameters,
    Stage2GenerationPolicy,
)
from oci.inference.post_extraction_scientific_policy import (
    PostExtractionScientificPolicy,
)
from oci.inference.all_evidence_post_extraction_review import (
    CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
)
from oci.inference.scoped_embedding_cache import SharedEmbeddingCache
from oci.inference.production_text_preparation import (
    TextPreparationOptions,
    prepare_modeling_cohort,
    stable_file_sha256,
)
from oci.models.concept_embedding_utils import chunk_text_words
from tests.test_post_extraction_scientific_policy import (
    _mapping as _post_extraction_policy_mapping,
)
from tests.cluster_local_embedding_test_support import (
    cluster_local_embedding_config,
)


def _digest(label: str) -> str:
    return identity_sha256({"label": label})


def _resource_safety(
    *,
    maximum_allocation_fraction: float = 0.8,
    minimum_headroom_bytes: int = 6 * GIB,
    minimum_multi_device_throughput_ratio: float = 1.4,
) -> ResourcePerformanceSafetyPolicy:
    return ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=maximum_allocation_fraction,
        gpu_minimum_headroom_bytes=minimum_headroom_bytes,
        minimum_multi_device_throughput_ratio=(minimum_multi_device_throughput_ratio),
        maximum_coordination_proof_overhead_ratio=0.25,
        maximum_ordinary_read_amplification=1.75,
        minimum_benchmark_repetitions_per_scope=3,
        read_counter_source="logical_read_bytes",
        fail_on_external_gpu_occupants=True,
    )


def _forest_operational(
    cpu_budget: int = 1,
) -> StrictCausalForestOperationalSpec:
    return StrictCausalForestOperationalSpec(
        requested_host_cpu_budget=cpu_budget,
        verbose=0,
        use_ray=False,
        ray_remote_func_options=None,
    )


def _forest_spec() -> StrictCausalForestSpec:
    return StrictCausalForestSpec.from_mapping(
        {
            "implementation": "econml.dml.CausalForestDML",
            "tune_model": False,
            "featurizer": None,
            "treatment_featurizer": None,
            "discrete_outcome": False,
            "discrete_treatment": True,
            "categories": "auto",
            "crossfit": {
                "implementation": "sklearn.model_selection.StratifiedKFold",
                "n_splits": 2,
                "shuffle": True,
                "random_seed": 17,
            },
            "mc_iters": None,
            "mc_agg": "mean",
            "drate": True,
            "n_estimators": 80,
            "criterion": "mse",
            "max_depth": None,
            "min_samples_split": 10,
            "min_samples_leaf": 6,
            "min_weight_fraction_leaf": 0.0,
            "min_var_fraction_leaf": None,
            "min_var_leaf_on_val": False,
            "max_features": 0.7,
            "min_impurity_decrease": 0.0,
            "max_samples": 0.45,
            "min_balancedness_tol": 0.45,
            "honest": True,
            "inference": True,
            "fit_intercept": True,
            "subforest_size": 4,
            "random_seed": 17,
            "allow_missing": False,
            "treatment_model": {
                "implementation": "sklearn.ensemble.RandomForestClassifier",
                "n_estimators": 40,
                "criterion": "gini",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 6,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "oob_score": False,
                "random_seed": 17,
                "warm_start": False,
                "class_weight": None,
                "ccp_alpha": 0.0,
                "max_samples": None,
                "monotonic_cst": None,
            },
            "outcome_model": {
                "implementation": "sklearn.ensemble.RandomForestRegressor",
                "n_estimators": 40,
                "criterion": "squared_error",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 6,
                "min_weight_fraction_leaf": 0.0,
                "max_features": 1.0,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "oob_score": False,
                "random_seed": 17,
                "warm_start": False,
                "ccp_alpha": 0.0,
                "max_samples": None,
                "monotonic_cst": None,
            },
        }
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


def _generation_policy() -> Stage2GenerationPolicy:
    def parameters(*, thinking_enabled: bool) -> Stage2GenerationParameters:
        return Stage2GenerationParameters(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            seed=42,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_tokens=25_000,
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
            thinking_token_budget=5_000 if thinking_enabled else 0,
            transport_max_retries=0,
            schema_repair_attempts=1,
        )

    selector = parameters(thinking_enabled=True)
    extraction = parameters(thinking_enabled=False)
    return Stage2GenerationPolicy(
        interpret_architecture_chunk=selector,
        consolidate_architecture_candidates=selector,
        audit_architecture_coverage=selector,
        plan_cross_architecture_integration=selector,
        integrate_cross_architecture_candidates=selector,
        audit_rejected_candidates=selector,
        define_one_extraction_feature=extraction,
        feature_proposal_review=selector,
        patient_feature_extraction=extraction,
    )


def _post_extraction_policy() -> PostExtractionScientificPolicy:
    return PostExtractionScientificPolicy.from_mapping(_post_extraction_policy_mapping())


def _scientific_spec() -> ScientificWorkflowSpec:
    return ScientificWorkflowSpec(
        columns=WorkflowColumns(
            unit_id="person",
            text="narrative",
            treatment="therapy",
            outcome="response",
        ),
        clinical_question="Configured treatment effect question",
        architecture_profiles={
            family: {
                "profile": family,
                "enabled": True,
                **(
                    {
                        "producer_configuration": {
                            "semantic_member_batch_size": 3,
                        }
                    }
                    if family == "whole_cohort_embeddings"
                    else {}
                ),
                **(
                    {
                        "producer_configuration": (
                            cluster_local_embedding_config().as_dict()
                        )
                    }
                    if family == "cluster_local_embeddings"
                    else {}
                ),
            }
            for family in EVIDENCE_FAMILIES
        },
        text_windows=LosslessTextWindowSpec(
            complete_page_core_chars=73,
            complete_page_context_chars=8,
            complete_page_max_chars=89,
            reconciliation_fan_in=5,
            embedding_chunk_size_words=41,
            embedding_chunk_overlap_words=9,
            embedding_max_chunks=10_000,
            embedding_chunk_selection="last",
            embedding_max_seq_length=384,
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
        ),
        stage2_prompt_protocol=Stage2PromptProtocolSpec(
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
            final_upstream_meta_inner_folds=5,
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
        ),
        post_extraction_causal_review=PostExtractionCausalReviewSpec(
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
        ),
        max_candidate_variables=12,
        causal_estimator=_forest_spec(),
        preprocessing=TextPreprocessingSpec(
            empty_text_policy="marker",
            repeated_character_policy="marker",
            repeated_character_threshold=777,
            source_text_temporally_valid_by_design=True,
        ),
        folds=FoldReviewSpec(
            outer_folds=5,
            review_rounds=2,
            initial_training_partitions=3,
            interaction_inner_folds=4,
            tfidf_nested_calibration_folds=4,
        ),
        seed=29,
        seed_policy="canonical_group_sha256_v1",
        prompt_identities={},
        estimand=BINARY_PROBABILITY_DIFFERENCE,
        compatibility_version=PORTABLE_SPEC_VERSION,
    )


def test_scientific_and_text_geometry_specs_have_no_implicit_defaults() -> None:
    for specification in (
        TextPreprocessingSpec,
        FoldReviewSpec,
        LosslessTextWindowSpec,
        Stage2PromptProtocolSpec,
        HierarchyWireBudgetSpec,
        PostExtractionCausalReviewSpec,
        StrictCausalForestSpec,
        SentenceEmbeddingEncoderSpec,
        CompletePagingGeometry,
        ScientificWorkflowSpec,
    ):
        defaulted = [
            field.name
            for field in fields(specification)
            if field.default is not MISSING or field.default_factory is not MISSING
        ]
        assert defaulted == []


def test_outer_folds_and_review_partition_counts_are_independent_configuration():
    folds = FoldReviewSpec(
        outer_folds=3,
        review_rounds=2,
        initial_training_partitions=2,
        interaction_inner_folds=3,
        tfidf_nested_calibration_folds=3,
    )
    assert folds.inner_partitions == 4
    assert folds.logical_context_count == 3 * (1 + 4 + 2)
    baseline = _scientific_spec()
    spec = replace(
        baseline,
        folds=folds,
        stage2_prompt_protocol=replace(
            baseline.stage2_prompt_protocol,
            final_upstream_meta_inner_folds=folds.inner_partitions,
        ),
    )
    assert spec.identity_payload()["folds"]["outer_folds"] == 3
    assert spec.identity_payload()["folds"]["inner_partitions"] == 4


def test_run_control_strict_round_trip_and_scientific_exclusion(
    tmp_path: Path,
) -> None:
    control = RunControl(
        resume=True,
        stop_after="handoff_validation",
        adopt_checkpoints=(
            tmp_path / "prepared",
            tmp_path / "embedding",
        ),
        log_level="warning",
        validation_depth="fresh_terminal_audit",
    )
    assert control.log_level == "WARNING"
    assert RunControl.from_mapping(control.as_dict()) == control
    path = tmp_path / "run-control.json"
    path.write_text(
        json.dumps(control.as_dict(), sort_keys=True),
        encoding="utf-8",
    )
    assert RunControl.from_json(path) == control
    assert control.as_dict()["schema_version"] == RUN_CONTROL_VERSION
    assert "run_control" not in _scientific_spec().identity_payload()


@pytest.mark.parametrize(
    ("update", "error"),
    (
        ({"resume": 1}, "boolean"),
        ({"stop_after": ""}, "nonempty"),
        (
            {"adopt_checkpoints": ["same", "same"]},
            "duplicated",
        ),
        ({"log_level": "verbose"}, "log level"),
        ({"validation_depth": "shallow"}, "validation depth"),
    ),
)
def test_run_control_rejects_invalid_values(
    update: dict[str, object],
    error: str,
) -> None:
    payload = RunControl().as_dict()
    payload.update(update)
    with pytest.raises((TypeError, ValueError), match=error):
        RunControl.from_mapping(payload)

    missing = dict(RunControl().as_dict())
    missing.pop("schema_version")
    with pytest.raises(ValueError, match="every field exactly"):
        RunControl.from_mapping(missing)
    extra = {**RunControl().as_dict(), "worker_pid": 12}
    with pytest.raises(ValueError, match="every field exactly"):
        RunControl.from_mapping(extra)


def test_scientific_identity_changes_for_window_config_not_deployment_metadata(
    tmp_path: Path,
) -> None:
    spec = _scientific_spec()
    profile_a = DeploymentProfile(
        dataset_path=tmp_path / "a" / "cohort.parquet",
        durable_artifact_root=tmp_path / "a" / "artifacts",
        scratch_root=tmp_path / "a" / "scratch",
        embedding_model_locator=tmp_path / "a" / "embed",
        htr_model_locator=tmp_path / "a" / "htr",
        stage1_profile_locator=tmp_path / "a" / "stage1.json",
        query_profile_locator=tmp_path / "a" / "query.json",
        embedding_batch_size=17,
        cluster_preflight_parquet_compression="zstd",
        resource_performance_safety=_resource_safety(),
        forest_operational=_forest_operational(3),
        stage1_execution=Stage1ExecutionProfile(
            resource_kind="accelerator",
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
        devices=("cuda:7",),
        cpu_budget=3,
        response_concurrency=5,
    )
    profile_b = replace(
        profile_a,
        dataset_path=tmp_path / "relocated" / "cohort.parquet",
        durable_artifact_root=tmp_path / "relocated" / "artifacts",
        scratch_root=tmp_path / "relocated" / "scratch",
        devices=("cpu",),
        embedding_batch_size=3,
        stage1_execution=Stage1ExecutionProfile(
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
        resource_performance_safety=_resource_safety(
            maximum_allocation_fraction=0.7,
            minimum_headroom_bytes=7 * GIB,
            minimum_multi_device_throughput_ratio=1.6,
        ),
        forest_operational=_forest_operational(1),
        cpu_budget=1,
        response_concurrency=1,
    )
    assert spec.scientific_sha256 == _scientific_spec().scientific_sha256
    assert profile_a.devices != profile_b.devices
    assert (
        profile_a.resource_performance_safety.content_sha256
        != profile_b.resource_performance_safety.content_sha256
    )
    runtime_a = compile_strict_causal_forest_runtime(
        scientific=spec,
        deployment=profile_a,
    )
    runtime_b = compile_strict_causal_forest_runtime(
        scientific=spec,
        deployment=profile_b,
    )
    assert runtime_a.scientific_identity() == runtime_b.scientific_identity()
    assert runtime_a.scientific_identity_sha256() == runtime_b.scientific_identity_sha256()
    assert runtime_a.operational_attestation() != runtime_b.operational_attestation()
    changed = replace(
        spec,
        text_windows=replace(
            spec.text_windows,
            complete_page_core_chars=71,
        ),
    )
    assert changed.scientific_sha256 != spec.scientific_sha256


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("prompt_policy", "authenticated_model_prompt_name"),
        ("output_value", "token_embeddings"),
        ("precision", "int8"),
        ("convert_to_numpy", False),
        ("convert_to_tensor", True),
        ("truncate_dim", 64),
        ("pooling_output_policy", "multiprocess"),
        ("model_dtype", "bfloat16"),
        ("stored_array_dtype", "float64"),
        ("zero_vector_policy", "preserve"),
    ),
)
def test_embedding_encoder_controls_are_closed_scientific_settings(
    field_name: str,
    changed_value: object,
) -> None:
    spec = _scientific_spec()
    encoder = spec.text_windows.embedding_encoder
    changes = {field_name: changed_value}
    if field_name == "prompt_policy":
        changes["prompt_name"] = "query"
    if field_name in {
        "output_value",
        "precision",
        "convert_to_numpy",
        "convert_to_tensor",
        "truncate_dim",
        "pooling_output_policy",
        "stored_array_dtype",
    }:
        with pytest.raises(ValueError):
            replace(encoder, **changes)
        return
    changed_encoder = replace(encoder, **changes)
    changed = replace(
        spec,
        text_windows=replace(
            spec.text_windows,
            embedding_encoder=changed_encoder,
        ),
    )
    assert changed.scientific_sha256 != spec.scientific_sha256


def test_typed_deployment_requires_every_operational_safety_field(
    tmp_path: Path,
) -> None:
    profile = DeploymentProfile(
        dataset_path=tmp_path / "cohort.parquet",
        durable_artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        embedding_model_locator=tmp_path / "embed",
        htr_model_locator=tmp_path / "htr",
        stage1_profile_locator=tmp_path / "stage1.json",
        query_profile_locator=tmp_path / "query.json",
        embedding_batch_size=5,
        cluster_preflight_parquet_compression="zstd",
        resource_performance_safety=_resource_safety(),
        forest_operational=_forest_operational(),
        stage1_execution=Stage1ExecutionProfile(
            resource_kind="accelerator",
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
    )
    payload = asdict(profile)
    reopened = DeploymentProfile.from_mapping(payload)
    assert reopened.resource_performance_safety == profile.resource_performance_safety
    assert (
        payload["resource_performance_safety"]["schema_version"]
        == RESOURCE_PERFORMANCE_SAFETY_VERSION
    )

    missing_deployment_field = dict(payload)
    missing_deployment_field.pop("cpu_budget")
    with pytest.raises(ValueError, match="every field exactly"):
        DeploymentProfile.from_mapping(missing_deployment_field)

    missing_execution_field = dict(payload)
    missing_execution_field["stage1_execution"] = dict(
        payload["stage1_execution"]
    )
    missing_execution_field["stage1_execution"].pop("scope_workers_per_device")
    with pytest.raises(ValueError, match="configure every field exactly"):
        DeploymentProfile.from_mapping(missing_execution_field)

    missing_safety_field = dict(payload)
    missing_safety_field["resource_performance_safety"] = dict(
        payload["resource_performance_safety"]
    )
    missing_safety_field["resource_performance_safety"].pop("gpu_minimum_headroom_bytes")
    with pytest.raises(ValueError, match="configure every field"):
        DeploymentProfile.from_mapping(missing_safety_field)


def test_nsclc_page_sizes_are_configuration_only_not_production_literals() -> None:
    repository = Path(__file__).resolve().parents[1]
    production_sources = tuple((repository / "oci").rglob("*.py")) + tuple(
        (repository / "scripts").rglob("*.py")
    )
    benchmark_literals = ("13488", "13_488", "14000", "14_000")

    offenders = {
        path.relative_to(repository).as_posix()
        for path in production_sources
        if any(literal in path.read_text(encoding="utf-8") for literal in benchmark_literals)
    }

    assert offenders == set()


def test_measured_stage1_execution_requires_reopenable_evidence_locators(
    tmp_path: Path,
) -> None:
    measured = Stage1ExecutionProfile(
        resource_kind="accelerator",
        device_count=2,
        scope_workers_per_device=2,
        executor_mode="persistent_slots",
        selection_method="measured_role_neutral_benchmark_v1",
        selected_candidate="measured-x2",
        benchmark_result_sha256="a" * 64,
        benchmark_result_locator=(
            tmp_path / "benchmark-result.json"
        ).resolve(),
        benchmark_workload_deployment_sha256="b" * 64,
        benchmark_workload_deployment_locator=(
            tmp_path / "workload-deployment.json"
        ).resolve(),
    )
    assert Stage1ExecutionProfile.from_mapping(
        asdict(measured)
    ) == measured
    json.dumps(measured.as_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="locators and hashes"):
        replace(measured, benchmark_workload_deployment_locator=None)
    with pytest.raises(ValueError, match="cannot claim benchmark"):
        replace(measured, selection_method="operator_configured")


def test_scientific_mapping_requires_text_window_and_forest_configuration() -> None:
    payload = _scientific_spec().identity_payload()
    # identity_payload resolves the estimand object, whereas from_mapping accepts
    # its registered name. Reconstruct a source-shape mapping for this gate.
    payload["estimand"] = _scientific_spec().estimand
    payload["compatibility_version"] = payload.pop("schema_version")
    payload["folds"] = {
        key: value
        for key, value in payload["folds"].items()
        if key not in {"inner_partitions", "logical_context_count"}
    }
    missing_windows = dict(payload)
    missing_windows.pop("text_windows")
    with pytest.raises(ValueError, match="text_windows"):
        ScientificWorkflowSpec.from_mapping(missing_windows)
    missing_stage2_protocol = dict(payload)
    missing_stage2_protocol.pop("stage2_prompt_protocol")
    with pytest.raises(ValueError, match="stage2_prompt_protocol"):
        ScientificWorkflowSpec.from_mapping(missing_stage2_protocol)
    missing_causal_review = dict(payload)
    missing_causal_review.pop("post_extraction_causal_review")
    with pytest.raises(ValueError, match="post_extraction_causal_review"):
        ScientificWorkflowSpec.from_mapping(missing_causal_review)
    missing_forest = dict(payload)
    missing_forest.pop("causal_estimator")
    with pytest.raises(ValueError, match="causal_estimator"):
        ScientificWorkflowSpec.from_mapping(missing_forest)
    incomplete_forest = dict(payload)
    incomplete_forest["causal_estimator"] = dict(payload["causal_estimator"])
    incomplete_forest["causal_estimator"]["treatment_model"] = dict(
        incomplete_forest["causal_estimator"]["treatment_model"]
    )
    incomplete_forest["causal_estimator"]["treatment_model"].pop("n_estimators")
    with pytest.raises(ValueError, match="n_estimators"):
        ScientificWorkflowSpec.from_mapping(incomplete_forest)
    missing_seed = dict(payload)
    missing_seed.pop("seed")
    with pytest.raises(ValueError, match="seed"):
        ScientificWorkflowSpec.from_mapping(missing_seed)
    missing_selection = dict(payload)
    missing_selection["text_windows"] = dict(payload["text_windows"])
    missing_selection["text_windows"].pop("embedding_chunk_selection")
    with pytest.raises(ValueError, match="embedding_chunk_selection"):
        ScientificWorkflowSpec.from_mapping(missing_selection)


def test_strict_forest_fixes_tuning_and_all_result_changing_settings() -> None:
    spec = _scientific_spec()
    baseline = spec.scientific_sha256
    changes = (
        replace(spec.causal_estimator, max_depth=7),
        replace(spec.causal_estimator, subforest_size=8),
        replace(
            spec.causal_estimator,
            treatment_model=replace(
                spec.causal_estimator.treatment_model,
                n_estimators=52,
                max_depth=9,
                min_samples_leaf=7,
                max_features=0.8,
            ),
        ),
        replace(
            spec.causal_estimator,
            outcome_model=replace(
                spec.causal_estimator.outcome_model,
                n_estimators=52,
                max_depth=9,
                min_samples_leaf=7,
                max_features="sqrt",
            ),
        ),
    )
    for changed_estimator in changes:
        changed = replace(
            spec,
            causal_estimator=changed_estimator,
        )
        assert changed.scientific_sha256 != baseline

    with pytest.raises(ValueError, match="tune_model=false"):
        replace(spec.causal_estimator, tune_model=True)
    with pytest.raises(ValueError, match="divisible by subforest_size"):
        replace(spec.causal_estimator, subforest_size=3)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("device", "cuda:3"),
        ("cache_dir", "/relocated/cache"),
        ("server_url", "https://endpoint.invalid/v1"),
        ("worker_count", 9),
    ),
)
def test_architecture_profiles_reject_deployment_metadata(
    field: str,
    value: object,
) -> None:
    spec = _scientific_spec()
    profiles = {name: dict(profile) for name, profile in spec.architecture_profiles.items()}
    profiles["hierarchical_transformer"][field] = value

    with pytest.raises(ValueError, match="deployment"):
        replace(spec, architecture_profiles=profiles)


def test_stage2_prompt_protocol_is_scientific_and_has_no_hidden_defaults():
    spec = _scientific_spec()
    changed = replace(
        spec,
        stage2_prompt_protocol=replace(
            spec.stage2_prompt_protocol,
            hierarchical_max_atoms_per_chunk=1,
        ),
    )
    assert changed.scientific_sha256 != spec.scientific_sha256
    assert (
        changed.identity_payload()["stage2_prompt_protocol"]["hierarchical_max_atoms_per_chunk"]
        == 1
    )
    changed_generation_policy = replace(
        spec,
        stage2_prompt_protocol=replace(
            spec.stage2_prompt_protocol,
            generation_policy=replace(
                spec.stage2_prompt_protocol.generation_policy,
                feature_proposal_review=replace(
                    spec.stage2_prompt_protocol.generation_policy.feature_proposal_review,
                    temperature=0.1,
                ),
            ),
        ),
    )
    assert changed_generation_policy.scientific_sha256 != spec.scientific_sha256
    with pytest.raises(ValueError, match="model_context_window_tokens"):
        replace(
            spec.stage2_prompt_protocol,
            model_context_window_tokens=spec.stage2_prompt_protocol.proposal_max_tokens,
        )
    changed_wire_budget = replace(
        spec,
        stage2_prompt_protocol=replace(
            spec.stage2_prompt_protocol,
            hierarchy_wire_budget=replace(
                spec.stage2_prompt_protocol.hierarchy_wire_budget,
                max_generated_list_items=12,
            ),
        ),
    )
    assert changed_wire_budget.scientific_sha256 != spec.scientific_sha256
    changed_extraction_protocol = replace(
        spec,
        stage2_prompt_protocol=replace(
            spec.stage2_prompt_protocol,
            extraction_grouping_strategy="clinical_domain",
            extraction_prompt_version="configured_prompt_revision",
        ),
    )
    assert changed_extraction_protocol.scientific_sha256 != spec.scientific_sha256
    incomplete_protocol = spec.stage2_prompt_protocol.as_dict()
    incomplete_protocol.pop("extraction_grouping_strategy")
    with pytest.raises(ValueError, match="explicitly and exactly"):
        Stage2PromptProtocolSpec.from_mapping(incomplete_protocol)
    with pytest.raises(ValueError, match="complete_paged_v1"):
        replace(
            spec.stage2_prompt_protocol,
            extraction_context_strategy="prefix_only",
        )
    with pytest.raises(ValueError, match="proposal_max_tokens conflicts"):
        replace(
            spec.stage2_prompt_protocol,
            proposal_max_tokens=(spec.stage2_prompt_protocol.proposal_max_tokens + 1),
        )
    mismatched_policy = replace(
        spec.stage2_prompt_protocol.generation_policy,
        patient_feature_extraction=replace(
            spec.stage2_prompt_protocol.generation_policy.patient_feature_extraction,
            max_tokens=(spec.stage2_prompt_protocol.extraction_max_tokens + 1),
        ),
    )
    with pytest.raises(ValueError, match="extraction_max_tokens conflicts"):
        replace(
            spec.stage2_prompt_protocol,
            generation_policy=mismatched_policy,
        )
    mismatched_thinking = replace(
        spec.stage2_prompt_protocol.generation_policy,
        feature_proposal_review=replace(
            spec.stage2_prompt_protocol.generation_policy.feature_proposal_review,
            thinking_token_budget=(spec.stage2_prompt_protocol.selector_thinking_token_budget - 1),
        ),
    )
    with pytest.raises(
        ValueError,
        match="selector_thinking_token_budget conflicts",
    ):
        replace(
            spec.stage2_prompt_protocol,
            generation_policy=mismatched_thinking,
        )
    incomplete_budget = asdict(spec.stage2_prompt_protocol.hierarchy_wire_budget)
    incomplete_budget.pop("max_generated_list_items")
    with pytest.raises(ValueError, match="hierarchy_wire_budget keys differ"):
        HierarchyWireBudgetSpec.from_mapping(incomplete_budget)


def test_final_oof_bank_folds_must_match_derived_stage1_inner_partitions():
    spec = _scientific_spec()
    assert (
        spec.stage2_prompt_protocol.final_upstream_meta_inner_folds
        == spec.folds.initial_training_partitions + spec.folds.review_rounds
    )
    with pytest.raises(
        ValueError,
        match="assembled from authenticated exact-inner transforms",
    ):
        replace(
            spec,
            stage2_prompt_protocol=replace(
                spec.stage2_prompt_protocol,
                final_upstream_meta_inner_folds=(
                    spec.stage2_prompt_protocol.final_upstream_meta_inner_folds - 1
                ),
            ),
        )


def test_causal_review_thresholds_are_scientific_and_exactly_configured():
    spec = _scientific_spec()
    changed = replace(
        spec,
        post_extraction_causal_review=replace(
            spec.post_extraction_causal_review,
            minimum_score_improvement=0.01,
        ),
    )
    assert changed.scientific_sha256 != spec.scientific_sha256
    changed_nested_policy = replace(
        spec,
        post_extraction_causal_review=replace(
            spec.post_extraction_causal_review,
            scientific_policy=replace(
                spec.post_extraction_causal_review.scientific_policy,
                extraction_quality=replace(
                    spec.post_extraction_causal_review.scientific_policy.extraction_quality,
                    minimum_coverage=0.06,
                ),
            ),
        ),
    )
    assert changed_nested_policy.scientific_sha256 != spec.scientific_sha256
    payload = spec.post_extraction_causal_review.as_dict()
    payload.pop("feature_bank_preservation_tolerance")
    with pytest.raises(ValueError, match="explicitly and exactly"):
        PostExtractionCausalReviewSpec.from_mapping(payload)
    changed_policy = replace(
        spec,
        post_extraction_causal_review=replace(
            spec.post_extraction_causal_review,
            upstream_review_policy=CONDITIONAL_CONTEXT_AND_GATE_REVIEW_POLICY,
        ),
    )
    assert changed_policy.scientific_sha256 != spec.scientific_sha256
    incomplete_policy = spec.post_extraction_causal_review.as_dict()
    incomplete_policy.pop("upstream_review_policy")
    with pytest.raises(ValueError, match="explicitly and exactly"):
        PostExtractionCausalReviewSpec.from_mapping(incomplete_policy)
    with pytest.raises(ValueError, match="explicitly registered"):
        replace(
            spec.post_extraction_causal_review,
            upstream_review_policy="implicit_or_unknown",
        )


def test_stage2_endpoint_requires_a_local_tokenizer_locator(tmp_path: Path) -> None:
    profile = DeploymentProfile(
        dataset_path=tmp_path / "cohort.parquet",
        durable_artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        embedding_model_locator=tmp_path / "embed",
        htr_model_locator=tmp_path / "htr",
        stage1_profile_locator=tmp_path / "stage1.json",
        query_profile_locator=tmp_path / "query.json",
        embedding_batch_size=1,
        cluster_preflight_parquet_compression="zstd",
        resource_performance_safety=_resource_safety(),
        forest_operational=_forest_operational(),
        stage1_execution=Stage1ExecutionProfile(
            resource_kind="accelerator",
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
        endpoint="https://model.example/v1",
        endpoint_model="exact/model",
        stage2_tokenizer_locator=tmp_path / "tokenizer",
    )
    assert profile.stage2_tokenizer_locator == tmp_path / "tokenizer"
    with pytest.raises(ValueError, match="stage2_tokenizer_locator"):
        replace(profile, stage2_tokenizer_locator=None)


def _compatibility() -> ArtifactCompatibility:
    return ArtifactCompatibility(
        dataset_identity=_digest("dataset"),
        split_identity=_digest("split"),
        row_order_identity=_digest("rows"),
        model_identities={"embed": _digest("embed")},
        prompt_identities={"extract": _digest("prompt")},
        configuration_identity=_digest("config"),
        seed_identity=_digest("seed"),
        producer_code_identity=_digest("producer"),
        runtime_compatibility_class="python-posix-test-v1",
    )


def _portable_artifact(root: Path):
    root.mkdir()
    (root / "values.bin").write_bytes(b"authenticated values")
    return publish_portable_artifact(
        root=root,
        artifact_kind="prepared_cohort",
        artifact_schema="prepared_test_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(),
        payload_paths=("values.bin",),
    )


def test_portable_artifact_relocation_adoption_and_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    original = _portable_artifact(tmp_path / "original")
    relocated = relocate_portable_artifact(
        source=original.root,
        target_root=tmp_path / "different" / "relocated",
    )
    assert relocated.artifact_id == original.artifact_id
    request_id = _digest("consumer request")
    first = adopt_checkpoint(
        source=relocated.root,
        attestation_root=tmp_path / "adoptions",
        consumer_request_sha256=request_id,
        expected_kind="prepared_cohort",
        expected_compatibility_key=original.compatibility_key,
        expected_upstream_artifact_ids=(),
    )
    second = adopt_checkpoint(
        source=relocated.root,
        attestation_root=tmp_path / "adoptions",
        consumer_request_sha256=request_id,
        expected_kind="prepared_cohort",
        expected_compatibility_key=original.compatibility_key,
        expected_upstream_artifact_ids=(),
    )
    assert first == second
    (relocated.root / "values.bin").write_bytes(b"tampered values")
    with pytest.raises(ValueError, match="changed"):
        validate_portable_artifact(relocated.root)


def test_process_authenticated_adoption_handle_rejects_later_mutation(
    tmp_path: Path,
) -> None:
    artifact = _portable_artifact(tmp_path / "handled")
    adopted = adopt_checkpoint(
        source=artifact.root,
        attestation_root=tmp_path / "handle_adoptions",
        consumer_request_sha256=_digest("handled consumer"),
        expected_kind="prepared_cohort",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(),
        validated_artifact=artifact,
    )
    assert adopted["producer_artifact_id"] == artifact.artifact_id

    (artifact.root / "values.bin").write_bytes(b"mutated after authentication")
    with pytest.raises(RuntimeError, match="changed after full-byte authentication"):
        adopt_checkpoint(
            source=artifact.root,
            attestation_root=tmp_path / "other_adoptions",
            consumer_request_sha256=_digest("other handled consumer"),
            validated_artifact=artifact,
        )


def test_portable_phase_binding_is_path_neutral_and_relocatable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "phase_artifact"
    payload = root / "prepared" / "modeling_cohort.parquet"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"prepared cohort bytes")
    audit = root / "prepared" / "preparation_manifest.json"
    audit.write_text('{"status":"complete"}', encoding="utf-8")
    external = tmp_path / "external_dataset.parquet"
    external.write_bytes(b"source bytes")
    artifact = publish_portable_artifact(
        root=root,
        artifact_kind="prepared_cohort",
        artifact_schema="prepared_phase_test_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(),
        payload_paths=(
            "prepared/modeling_cohort.parquet",
            "prepared/preparation_manifest.json",
        ),
        workflow_phase="input_preparation",
        workflow_phase_result={
            "source": {"path": str(external.resolve())},
            "output": {"path": str(payload.resolve())},
            "terminal_files": [
                str(payload.resolve()),
                str(audit.resolve()),
            ],
        },
    )
    materialized = materialize_portable_phase(
        artifact,
        expected_phase="input_preparation",
    )
    assert materialized["result"]["output"]["path"] == str(payload.resolve())
    assert materialized["result"]["source"]["path"] is None

    relocated = relocate_portable_artifact(
        source=artifact.root,
        target_root=tmp_path / "relocated_phase_artifact",
    )
    rebound = materialize_portable_phase(
        relocated,
        expected_phase="input_preparation",
    )
    assert relocated.artifact_id == artifact.artifact_id
    assert rebound["result"]["output"]["path"] == str(
        (relocated.root / "prepared/modeling_cohort.parquet").resolve()
    )
    with pytest.raises(ValueError, match="different workflow phase"):
        materialize_portable_phase(
            relocated,
            expected_phase="embedding_cache",
        )


def test_portable_artifact_rejects_extra_symlink_and_hardlink(tmp_path: Path) -> None:
    extra = _portable_artifact(tmp_path / "extra")
    (extra.root / "unregistered.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered"):
        validate_portable_artifact(extra.root)

    symlinked = _portable_artifact(tmp_path / "symlinked")
    (symlinked.root / "unregistered-link").symlink_to("values.bin")
    with pytest.raises(ValueError, match="symlink"):
        validate_portable_artifact(symlinked.root)

    hardlinked = _portable_artifact(tmp_path / "hardlinked")
    os.link(
        hardlinked.root / "values.bin",
        hardlinked.root / "unregistered-hardlink",
    )
    with pytest.raises(ValueError, match="hard link"):
        validate_portable_artifact(hardlinked.root)


def test_portable_reference_adopts_existing_complete_tree_without_copy(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "existing_terminal_payload"
    payload_root.mkdir()
    (payload_root / "table.parquet").write_bytes(b"terminal table bytes")
    nested = payload_root / "nested"
    nested.mkdir()
    (nested / "array.npy").write_bytes(b"terminal array bytes")
    artifact = publish_portable_reference_artifact(
        control_root=tmp_path / "portable_control",
        payload_root=payload_root,
        artifact_kind="prepared_cohort",
        artifact_schema="prepared_reference_test_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(),
        payload_paths=("table.parquet", "nested/array.npy"),
    )
    assert artifact.payload_root == payload_root.resolve()
    assert not (artifact.root / "table.parquet").exists()
    adopted = adopt_checkpoint(
        source=artifact.root,
        attestation_root=tmp_path / "reference_adoptions",
        consumer_request_sha256=_digest("reference consumer"),
        expected_kind="prepared_cohort",
        expected_compatibility_key=artifact.compatibility_key,
        expected_upstream_artifact_ids=(),
    )
    assert adopted["producer_artifact_id"] == artifact.artifact_id
    (payload_root / "nested/array.npy").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        validate_portable_artifact(artifact.root)


def test_portable_reference_reuses_same_process_authenticated_stat_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_root = tmp_path / "same_process_payload"
    payload_root.mkdir()
    payload = payload_root / "values.npy"
    payload.write_bytes(b"authenticated without a redundant publication read")
    state = os.lstat(payload)
    stat_identity = (
        int(state.st_dev),
        int(state.st_ino),
        int(state.st_mode),
        int(state.st_nlink),
        int(state.st_size),
        int(state.st_mtime_ns),
        int(state.st_ctime_ns),
    )
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    monkeypatch.setattr(
        portable_artifacts_module,
        "_safe_file_hash",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-process publication reread payload bytes")
        ),
    )
    artifact = publish_portable_reference_artifact(
        control_root=tmp_path / "same_process_control",
        payload_root=payload_root,
        artifact_kind="prepared_cohort",
        artifact_schema="same_process_reference_test_v1",
        compatibility=_compatibility(),
        upstream_artifact_ids=(),
        payload_paths=("values.npy",),
        expected_payload_identities={
            "values.npy": (digest, len(payload.read_bytes())),
        },
        process_authenticated_stat_inventory={
            "values.npy": stat_identity,
        },
    )
    assert artifact.payloads[0].sha256 == digest

    changed_root = tmp_path / "changed_same_process_payload"
    changed_root.mkdir()
    changed = changed_root / "values.npy"
    changed.write_bytes(b"before")
    changed_state = os.lstat(changed)
    changed_identity = (
        int(changed_state.st_dev),
        int(changed_state.st_ino),
        int(changed_state.st_mode),
        int(changed_state.st_nlink),
        int(changed_state.st_size),
        int(changed_state.st_mtime_ns),
        int(changed_state.st_ctime_ns),
    )
    changed.write_bytes(b"after!!")
    with pytest.raises(RuntimeError, match="payload changed"):
        publish_portable_reference_artifact(
            control_root=tmp_path / "changed_same_process_control",
            payload_root=changed_root,
            artifact_kind="prepared_cohort",
            artifact_schema="changed_same_process_reference_test_v1",
            compatibility=_compatibility(),
            upstream_artifact_ids=(),
            payload_paths=("values.npy",),
            expected_payload_identities={
                "values.npy": (
                    hashlib.sha256(b"before").hexdigest(),
                    len(b"before"),
                ),
            },
            process_authenticated_stat_inventory={
                "values.npy": changed_identity,
            },
        )


def test_legacy_preparation_migrates_but_v2_embedding_cache_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = WorkflowColumns(
        unit_id="record_key",
        text="complete_note",
        treatment="received_drug",
        outcome="response_flag",
    )
    preprocessing = TextPreprocessingSpec(
        empty_text_policy="marker",
        repeated_character_policy="marker",
        repeated_character_threshold=5,
        source_text_temporally_valid_by_design=True,
    )
    source = tmp_path / "arbitrary_source.parquet"
    pd.DataFrame(
        {
            columns.unit_id: ["r3", "r1", "r4", "r2"],
            columns.text: [
                "alpha beta gamma delta epsilon",
                "",
                "one two three four five six",
                "punctuation !!!!! should be marked",
            ],
            columns.treatment: [0, 1, 0, 1],
            columns.outcome: [1, 0, 0, 1],
        }
    ).to_parquet(source, index=False)
    source_sha256, source_size = stable_file_sha256(source)
    model_sha256 = _digest("legacy embedding model tree")
    compatibility = replace(
        _compatibility(),
        dataset_identity=source_sha256,
        row_order_identity=_digest("configured source row order"),
        model_identities={"embed": model_sha256},
    )

    def write_phase(
        phase: str,
        *,
        attempt: Path,
        result: dict,
    ) -> Path:
        phase_root = attempt.parent
        registrations = []
        for payload_path in sorted(
            (path for path in attempt.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(attempt).as_posix(),
        ):
            payload, size = stable_file_sha256(payload_path)
            registrations.append(
                {
                    "path": str(payload_path.resolve()),
                    "relative_path": payload_path.relative_to(attempt).as_posix(),
                    "sha256": payload,
                    "size_bytes": size,
                }
            )
        body = {
            "schema_version": "production_workflow_phase_manifest_v2",
            "status": "complete",
            "phase": phase,
            "request_sha256": _digest("legacy request"),
            "attempt_dir": str(attempt.resolve()),
            "result": result,
            "artifacts": registrations,
        }
        manifest = {**body, "content_sha256": identity_sha256(body)}
        manifest_path = phase_root / "complete_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        return manifest_path

    preparation_phase = tmp_path / "legacy_input_preparation"
    preparation_attempt = preparation_phase / "attempt"
    preparation_attempt.mkdir(parents=True)
    preparation_result = prepare_modeling_cohort(
        TextPreparationOptions(
            dataset_path=source,
            output_dir=preparation_attempt / "prepared",
            unit_id_column=columns.unit_id,
            text_column=columns.text,
            treatment_column=columns.treatment,
            outcome_column=columns.outcome,
            repeated_character_threshold=(preprocessing.repeated_character_threshold),
            empty_text_policy=preprocessing.empty_text_policy,
            repeated_character_policy=preprocessing.repeated_character_policy,
        )
    )
    prepared_path = preparation_attempt / "prepared" / "modeling_cohort.parquet"
    preparation_manifest_path = preparation_attempt / "prepared" / "preparation_manifest.json"
    prepared_frame = pd.read_parquet(
        prepared_path,
        columns=[
            columns.unit_id,
            columns.text,
            columns.treatment,
            columns.outcome,
        ],
    )
    prepared_expectation = LegacyPreparedMigrationExpectation(
        columns=columns,
        preprocessing=preprocessing,
        dataset_sha256=source_sha256,
        dataset_size_bytes=source_size,
        prepared_cohort_sha256=preparation_result["output"]["sha256"],
        prepared_projection_sha256=(
            legacy_migration_module._ordered_prepared_projection_sha256(prepared_frame)
        ),
        unit_id_order_sha256=legacy_migration_module._unit_id_order_sha256(
            prepared_frame[columns.unit_id].tolist()
        ),
        row_order_identity=compatibility.row_order_identity,
        expected_row_count=len(prepared_frame),
    )
    prepared_manifest = write_phase(
        "input_preparation",
        attempt=preparation_attempt,
        result={
            **preparation_result,
            "terminal_files": [
                str(prepared_path.resolve()),
                str(preparation_manifest_path.resolve()),
            ],
        },
    )
    prepared = migrate_legacy_terminal_phase_reference(
        manifest_path=prepared_manifest,
        expected_phase="input_preparation",
        control_root=tmp_path / "prepared_control",
        artifact_kind="prepared_cohort",
        artifact_schema="legacy_prepared_migration_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        typed_expectation=prepared_expectation,
        upstream_prepared_artifact=None,
    )
    assert prepared.payload_root.name == "attempt"
    assert prepared.phase_binding["phase"] == "input_preparation"
    prepared_migration = materialize_portable_phase(
        prepared,
        expected_phase="input_preparation",
    )[
        "result"
    ]["legacy_terminal_migration_identity"]
    assert prepared_migration["current_preparation_transform_replayed"] is True
    assert prepared_migration["source_text_temporal_validity_legacy_field_available"] is False
    with pytest.raises(ValueError, match="prepared cohort"):
        migrate_legacy_terminal_phase_reference(
            manifest_path=prepared_manifest,
            expected_phase="input_preparation",
            control_root=tmp_path / "wrong_count_control",
            artifact_kind="prepared_cohort",
            artifact_schema="legacy_prepared_migration_test_v1",
            compatibility=compatibility,
            upstream_artifact_ids=(),
            typed_expectation=replace(
                prepared_expectation,
                expected_row_count=1000,
            ),
            upstream_prepared_artifact=None,
        )
    with pytest.raises(ValueError, match="preprocessing policy"):
        migrate_legacy_terminal_phase_reference(
            manifest_path=prepared_manifest,
            expected_phase="input_preparation",
            control_root=tmp_path / "wrong_preprocessing_control",
            artifact_kind="prepared_cohort",
            artifact_schema="legacy_prepared_migration_test_v1",
            compatibility=compatibility,
            upstream_artifact_ids=(),
            typed_expectation=replace(
                prepared_expectation,
                preprocessing=replace(
                    preprocessing,
                    repeated_character_threshold=6,
                ),
            ),
            upstream_prepared_artifact=None,
        )
    with pytest.raises(ValueError, match="dataset/order request"):
        migrate_legacy_terminal_phase_reference(
            manifest_path=prepared_manifest,
            expected_phase="input_preparation",
            control_root=tmp_path / "wrong_row_order_control",
            artifact_kind="prepared_cohort",
            artifact_schema="legacy_prepared_migration_test_v1",
            compatibility=replace(
                compatibility,
                row_order_identity=_digest("different row order"),
            ),
            upstream_artifact_ids=(),
            typed_expectation=prepared_expectation,
            upstream_prepared_artifact=None,
        )

    cache_phase = tmp_path / "legacy_embedding_cache"
    cache_attempt = cache_phase / "attempt"
    relocated = cache_attempt / "relocated_cache"
    cache_dir = relocated / "embedding_cache"
    relocated_prepared = relocated / "prepared" / "modeling_cohort.parquet"
    cache_dir.mkdir(parents=True)
    relocated_prepared.parent.mkdir(parents=True)
    shutil.copyfile(prepared_path, relocated_prepared)
    chunk_configuration = {
        "chunk_size_words": 3,
        "chunk_overlap_words": 1,
        "max_chunks": 10,
        "chunk_selection": "last",
        "normalize_embeddings": False,
        "max_seq_length": 128,
    }
    texts = tuple(prepared_frame[columns.text].tolist())
    chunk_rows = [
        tuple(
            chunk_text_words(
                text,
                chunk_configuration["chunk_size_words"],
                chunk_configuration["chunk_overlap_words"],
                chunk_configuration["max_chunks"],
                chunk_configuration["chunk_selection"],
            )
        )
        for text in texts
    ]
    chunk_counts = [len(row) for row in chunk_rows]
    offsets = np.zeros(len(chunk_rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(chunk_counts, dtype=np.int64))
    with (cache_dir / "offsets.npy").open("xb") as handle:
        np.save(handle, offsets, allow_pickle=False)
    embeddings = np.arange(
        int(offsets[-1]) * 4,
        dtype=np.float32,
    ).reshape(int(offsets[-1]), 4)
    with (cache_dir / "chunk_embeddings.npy").open("xb") as handle:
        np.save(handle, embeddings, allow_pickle=False)
    (cache_dir / "chunk_texts.jsonl").write_text(
        "".join(
            json.dumps(
                {"chunks": list(chunks)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for chunks in chunk_rows
        ),
        encoding="utf-8",
    )
    companion_registrations = {
        name: {
            "sha256": stable_file_sha256(cache_dir / name)[0],
            "size_bytes": stable_file_sha256(cache_dir / name)[1],
        }
        for name in (
            "chunk_embeddings.npy",
            "chunk_texts.jsonl",
            "offsets.npy",
        )
    }
    ordered_text_sha256 = legacy_migration_module._ordered_text_sha256(
        text_column=columns.text,
        texts=texts,
    )
    legacy_builder_sha256 = _digest("accepted legacy cache builder")
    configuration_sha256 = identity_sha256(chunk_configuration)
    cache_configuration_sha256 = identity_sha256(
        {
            "schema_version": ("production_embedding_cache_configuration_identity_v1"),
            "sentence_model_name": "arbitrary-embedding-model",
            "chunk_configuration": chunk_configuration,
        }
    )
    token_counts = [max(1, len(chunk.split()) + 2) for chunks in chunk_rows for chunk in chunks]
    token_counts_sha256 = identity_sha256(token_counts)
    provenance = {
        "schema_version": ("production_arbitrary_cohort_embedding_cache_provenance_v2"),
        "builder_version": ("production_arbitrary_cohort_embedding_cache_builder_v2"),
        "builder_code_sha256": legacy_builder_sha256,
        "dataset": {
            "path": str(prepared_path.resolve()),
            "sha256": prepared_expectation.prepared_cohort_sha256,
            "size_bytes": stable_file_sha256(prepared_path)[1],
            "text_column": columns.text,
            "row_count": len(prepared_frame),
            "ordered_text_sha256": ordered_text_sha256,
        },
        "sentence_model_name": "arbitrary-embedding-model",
        "local_model": {
            "path": str((tmp_path / "unused_model_locator").resolve()),
            "tree_sha256": model_sha256,
            "file_count": 1,
            "directory_count": 1,
            "total_file_bytes": 1,
        },
        "chunk_configuration": chunk_configuration,
        "chunk_configuration_sha256": configuration_sha256,
        "cache_configuration_sha256": cache_configuration_sha256,
        "encoder_execution": {
            "device": "cpu",
            "batch_size": 2,
            "local_files_only": True,
            "trust_remote_code": False,
            "offline_environment": {},
            "socket_access_blocked": True,
        },
        "companion_cache_files": companion_registrations,
        "uncapped_total_chunks": int(offsets[-1]),
        "uncapped_chunk_counts_sha256": identity_sha256(chunk_counts),
        "chunk_cap_nonbinding": True,
        "semantic_truncation_allowed": False,
        "max_observed_token_count": max(token_counts),
        "ordered_token_counts_sha256": token_counts_sha256,
        "tokenizer_truncation_allowed": False,
        "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
        "partial_cache_reuse_allowed": False,
        "network_access_allowed": False,
        "symlinks_allowed": False,
        "executable_artifacts_allowed": False,
    }
    metadata = {
        "schema_version": ("production_arbitrary_cohort_embedding_cache_metadata_v2"),
        "sentence_model_name": "arbitrary-embedding-model",
        "hidden_size": 4,
        "num_samples": len(prepared_frame),
        "total_chunks": int(offsets[-1]),
        "chunk_counts": chunk_counts,
        **chunk_configuration,
        "effective_max_seq_length": 128,
        "chunking_mode": ("whitespace_word_chunks_tokenizer_verified_nontruncating_v2"),
        "actual_max_len": max(chunk_counts),
        "uncapped_total_chunks": int(offsets[-1]),
        "uncapped_chunk_counts_sha256": identity_sha256(chunk_counts),
        "chunk_cap_nonbinding": True,
        "semantic_truncation_allowed": False,
        "max_observed_token_count": max(token_counts),
        "ordered_token_counts_sha256": token_counts_sha256,
        "tokenizer_truncation_allowed": False,
        "storage_format": "variable_length_chunks",
        "dtype": "float32",
        "production_provenance": provenance,
        "production_provenance_sha256": identity_sha256(provenance),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    cache_files = {
        name: {
            "sha256": stable_file_sha256(cache_dir / name)[0],
            "size_bytes": stable_file_sha256(cache_dir / name)[1],
        }
        for name in (
            "chunk_embeddings.npy",
            "chunk_texts.jsonl",
            "metadata.json",
            "offsets.npy",
        )
    }
    provider_identity = {
        "provider": "spent_only_frozen_chunk_embedding_cache_v2",
        "embeddings_sha256": cache_files["chunk_embeddings.npy"]["sha256"],
        "chunk_texts_sha256": cache_files["chunk_texts.jsonl"]["sha256"],
        "metadata_sha256": cache_files["metadata.json"]["sha256"],
        "offsets_sha256": cache_files["offsets.npy"]["sha256"],
        "row_count": len(prepared_frame),
        "chunk_count": int(offsets[-1]),
        "cache_snapshot_authentication": "streamed_private_fd_sha256_v1",
        "chunk_text_storage": "private_fd_pread_lazy_row_decode_v1",
        "embeddings_path_backed": False,
        "private_snapshot_embedding_mmap": True,
        "future_row_text_decoded": False,
        "novel_text_encoding_allowed": False,
    }
    build_identity = {
        "schema_version": ("production_arbitrary_cohort_embedding_cache_result_v2"),
        "builder_version": ("production_arbitrary_cohort_embedding_cache_builder_v2"),
        "builder_code_sha256": legacy_builder_sha256,
        "cache_path": str(cache_dir.resolve()),
        "production_provenance_sha256": identity_sha256(provenance),
        "dataset_sha256": prepared_expectation.prepared_cohort_sha256,
        "ordered_text_sha256": ordered_text_sha256,
        "sentence_model_name": "arbitrary-embedding-model",
        "local_model_tree_sha256": model_sha256,
        "chunk_configuration_sha256": configuration_sha256,
        "cache_configuration_sha256": cache_configuration_sha256,
        "row_count": len(prepared_frame),
        "chunk_count": int(offsets[-1]),
        "hidden_size": 4,
        "cache_files": cache_files,
        "provider_identity": provider_identity,
        "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
        "offline_build": True,
    }
    relocation_attestation = relocated / "relocation_attestation.json"
    relocation_attestation.write_text('{"fixture":"attestation"}\n', encoding="utf-8")
    relocation_terminal = relocated / "complete_manifest.json"
    relocation_terminal.write_text('{"fixture":"terminal"}\n', encoding="utf-8")
    attestation_sha, _ = stable_file_sha256(relocation_attestation)
    terminal_sha, _ = stable_file_sha256(relocation_terminal)
    requested_chunk_configuration = {
        **chunk_configuration,
        "prompt_policy": "disabled",
        "prompt_name": None,
        "output_value": "sentence_embedding",
        "precision": "float32",
        "convert_to_numpy": True,
        "convert_to_tensor": False,
        "truncate_dim": None,
        "pooling_output_policy": "single_process_sentence_embedding_v1",
        "model_dtype": "float32",
        "stored_array_dtype": "float32",
        "zero_vector_policy": "reject",
    }
    cache_expectation = LegacyEmbeddingCacheMigrationExpectation(
        prepared=prepared_expectation,
        embedding_model_name="arbitrary-embedding-model",
        embedding_model_tree_sha256=model_sha256,
        chunk_configuration=requested_chunk_configuration,
        ordered_text_sha256=ordered_text_sha256,
        expected_chunk_count=int(offsets[-1]),
        expected_hidden_size=4,
        legacy_builder_code_sha256=legacy_builder_sha256,
    )
    cache_result = {
        "schema_version": "production_embedding_cache_phase_result_v1",
        "mode": "authenticated_relocation",
        "cache_path": str(cache_dir.resolve()),
        "prepared_cohort_path": str(relocated_prepared.resolve()),
        "row_count": len(prepared_frame),
        "cache_identity": {
            "schema_version": "production_embedding_cache_relocation_result_v2",
            "root": str(relocated.resolve()),
            "cache_dir": str(cache_dir.resolve()),
            "prepared_cohort_path": str(relocated_prepared.resolve()),
            "attestation_path": str(relocation_attestation.resolve()),
            "terminal_manifest_path": str(relocation_terminal.resolve()),
            "row_count": len(prepared_frame),
            "prepared_projection_sha256": (prepared_expectation.prepared_projection_sha256),
            "cache_build_identity": build_identity,
            "attestation_sha256": attestation_sha,
            "terminal_manifest_sha256": terminal_sha,
        },
        "terminal_files": [
            str(path.resolve())
            for path in sorted(
                (path for path in cache_attempt.rglob("*") if path.is_file()),
                key=lambda path: path.relative_to(cache_attempt).as_posix(),
            )
        ],
    }
    cache_manifest = write_phase(
        "embedding_cache",
        attempt=cache_attempt,
        result=cache_result,
    )
    with pytest.raises(ValueError, match="allowlisted frozen V5 producer"):
        migrate_legacy_terminal_phase_reference(
            manifest_path=cache_manifest,
            expected_phase="embedding_cache",
            control_root=tmp_path / "cache_control",
            artifact_kind="embedding_cache",
            artifact_schema="legacy_cache_migration_test_v1",
            compatibility=compatibility,
            upstream_artifact_ids=(prepared.artifact_id,),
            typed_expectation=cache_expectation,
            upstream_prepared_artifact=prepared,
        )
    # Field-shape similarity is never enough. Only the exact frozen V5
    # source/builder/model producer may derive omitted v2 encoder semantics.


def test_five_fold_context_plan_discovers_40_logical_and_35_physical() -> None:
    partitions = {
        fold: tuple(
            tuple(f"f{fold}-p{partition}-r{row}" for row in range(4)) for partition in range(1, 6)
        )
        for fold in range(1, 6)
    }
    heldout = {fold: tuple(f"f{fold}-heldout-{row}" for row in range(5)) for fold in range(1, 6)}
    contexts = derive_logical_context_plan(
        outer_training_partitions=partitions,
        outer_heldout_rows=heldout,
        architecture_identity=_digest("architecture"),
        target="all_ten_evidence_families",
        scientific_configuration_identity=_digest("scientific config"),
        global_seed=42,
        producer_identity=_digest("producer"),
        runtime_compatibility_class="python-posix-test-v1",
        review_rounds=2,
    )
    groups = group_equivalent_contexts(contexts)
    assert len(contexts) == 40
    assert len(groups) == 35
    duplicates = [group for group in groups if len(group.logical_contexts) == 2]
    assert len(duplicates) == 5
    assert all(
        group.canonical_owner.scope_id.endswith("_inner_005")
        and group.logical_contexts[1].scope_id.endswith("_hierarchy_epoch_001")
        for group in duplicates
    )
    assert all(
        len({context.scope_seed for context in group.logical_contexts}) == 1 for group in duplicates
    )
    physical_ids = {group.key.key: _digest(f"physical:{group.key.key}") for group in groups}
    family_ids = {
        group.key.key: {
            family: _digest(f"{group.key.key}:{family}") for family in EVIDENCE_FAMILIES
        }
        for group in groups
    }
    bindings = build_logical_binding_records(
        groups=groups,
        physical_artifact_ids=physical_ids,
        physical_family_artifact_ids=family_ids,
    )
    assert bindings["logical_context_count"] == 40
    assert bindings["physical_fit_count"] == 35
    assert bindings["deduplicated_fit_count"] == 5
    assert all(
        set(row["family_artifact_ids"]) == set(EVIDENCE_FAMILIES)
        for row in bindings["logical_bindings"]
    )


def test_legacy_classifier_rejects_partial_v5_and_migration_accounts_for_v4(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "v5"

    def write_terminal_candidate(phase: str) -> None:
        phase_root = partial / "phases" / phase
        attempt = phase_root / "attempt"
        attempt.mkdir(parents=True)
        payload = attempt / "payload.bin"
        payload_bytes = f"{phase}-payload".encode("utf-8")
        payload.write_bytes(payload_bytes)
        body = {
            "schema_version": "production_workflow_phase_manifest_v2",
            "status": "complete",
            "phase": phase,
            "request_sha256": _digest("partial-v5-request"),
            "attempt_dir": str(attempt.resolve()),
            "result": {},
            "artifacts": [
                {
                    "path": str(payload.resolve()),
                    "relative_path": payload.name,
                    "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    "size_bytes": len(payload_bytes),
                }
            ],
        }
        terminal = phase_root / "complete_manifest.json"
        terminal.write_text(
            json.dumps(
                {**body, "content_sha256": identity_sha256(body)},
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    for phase in ("input_preparation", "embedding_cache"):
        write_terminal_candidate(phase)
    (partial / "recovery" / "cluster_preflight_scope_inputs").mkdir(parents=True)
    incomplete_attempt = partial / "phases" / "stage1_preflight" / "attempt"
    incomplete_attempt.mkdir(parents=True)
    (incomplete_attempt / "effective_stage1_profile.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    classification = classify_legacy_workflow(partial)
    assert classification["prepared_cohort_candidate"] is True
    assert classification["embedding_cache_candidate"] is True
    assert (
        classification["preparation_and_cache_adoption_requires_current_full_byte_validator"]
        is True
    )
    assert classification["incomplete_preflight_categorically_rejected"] is True
    assert classification["clustered_preflight_directly_portable"] is False
    assert all(
        row["registered_payload_bytes_authenticated"] is False for row in classification["phases"]
    )

    marker_only = tmp_path / "marker_only"
    marker = marker_only / "phases" / "input_preparation" / "complete_manifest.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    marker_classification = classify_legacy_workflow(marker_only)
    assert marker_classification["prepared_cohort_candidate"] is False
    assert marker_classification["terminal_marker_presence_alone_is_not_a_candidate"] is True

    partitions = {
        fold: tuple(
            tuple(f"{fold}{partition}{row}" for row in range(3)) for partition in range(1, 6)
        )
        for fold in range(1, 6)
    }
    heldout = {fold: tuple(f"h{fold}{row}" for row in range(3)) for fold in range(1, 6)}
    contexts = derive_logical_context_plan(
        outer_training_partitions=partitions,
        outer_heldout_rows=heldout,
        architecture_identity=_digest("legacy architecture"),
        target="clustered_preflight",
        scientific_configuration_identity=_digest("legacy config"),
        global_seed=42,
        producer_identity=_digest("legacy producer"),
        runtime_compatibility_class="legacy-runtime-test",
        review_rounds=2,
    )
    legacy_root = tmp_path / "v4" / "cluster_preflight"
    legacy_root.mkdir(parents=True)
    audit = legacy_root / "cluster_feasibility_audit.json"
    request = legacy_root / "stage1_preflight_request.json"
    audit.write_bytes(b'{"complete":true}\n')
    request.write_bytes(b'{"complete":true}\n')

    def registration(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "relative_path": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    scope_records = []
    for index, context in enumerate(contexts):
        fit_values = [int(value) for value in context.fit_row_ids]
        heldout_values = list(context.heldout_row_ids)
        is_cumulative = context.purpose.startswith("cumulative_review_epoch_")
        legacy_fit_values = list(reversed(fit_values)) if is_cumulative else fit_values
        legacy_heldout_values = list(reversed(heldout_values)) if is_cumulative else heldout_values
        scope_records.append(
            {
                "canonical_index": index,
                "scope_id": context.scope_id,
                "scope_kind": context.purpose,
                "outer_fold": context.outer_fold,
                "inner_fold": None,
                "context_epoch": None,
                "provider_inner_fold": None,
                "fit_row_count": len(fit_values),
                "fit_row_order_fingerprint": (
                    legacy_migration_module._legacy_row_fingerprint(legacy_fit_values)
                ),
                "heldout_row_count": len(context.heldout_row_ids),
                "heldout_row_order_fingerprint": (
                    legacy_migration_module._legacy_row_fingerprint(legacy_heldout_values)
                ),
                "scope_record_sha256": _digest(f"record-{context.scope_id}"),
                "cluster_fit_identity_sha256": _digest(f"cluster-{context.scope_id}"),
            }
        )
    manifest_body = {
        "schema_version": "production_stage1_cluster_preflight_manifest_v1",
        "status": "complete",
        "artifact_version": "legacy-test-v1",
        "artifact_code_sha256": _digest("legacy code"),
        "root": str(legacy_root),
        "files": {
            "audit": registration(audit),
            "stage1_request": registration(request),
        },
        "bindings": {},
        "scope_records": scope_records,
    }
    manifest = {
        **manifest_body,
        "content_sha256": identity_sha256(manifest_body),
    }
    manifest_path = legacy_root / "cluster_preflight_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    migration = plan_legacy_v4_preflight_migration(
        manifest_path=manifest_path,
        logical_contexts=contexts,
        authenticate_registered_payload_bytes=True,
    )
    assert migration["logical_scope_count"] == 40
    assert migration["physical_fit_count"] == 35
    assert migration["deduplicated_group_count"] == 5
    assert migration["decision"] == "recompute_required"
    assert migration["recompute_physical_fit_count"] == 35
    assert "legacy_safe_kmeans_svd_state_payloads_absent" in migration["recompute_reason_codes"]
    assert migration["dependency_proof"]["registered_payload_bytes_freshly_authenticated"] is True
    assert migration["dependency_proof"]["requested_fit_row_orders_match_legacy_records"] is False
    assert migration["dependency_proof"]["all_dependencies_and_evidence_identities_proved"] is False
    assert (
        "legacy_cumulative_row_order_not_reusable_for_current_request"
        in migration["recompute_reason_codes"]
    )
    accounting = migration["accounting"]
    assert len(accounting["superseded_duplicate_outputs"]) == 5
    assert accounting["source_tree_mutated"] is False
    assert accounting["legacy_payload_copies_materialized"] is False
    assert (
        sum(
            row["legacy_order_disposition"] == "exact_request_match"
            for row in accounting["logical_bindings"]
        )
        == 30
    )
    assert (
        sum(
            row["legacy_order_disposition"] == "cumulative_historical_order_not_reusable"
            for row in accounting["logical_bindings"]
        )
        == 10
    )
    assert (
        sum(
            row["legacy_owner_order_reusable_for_current_fit"]
            for row in accounting["physical_records"]
        )
        == 30
    )
    assert all(
        row["canonical_owner_scope_seed"]
        == next(
            context.scope_seed
            for context in contexts
            if context.scope_id == row["canonical_owner_scope_id"]
        )
        and row["canonical_fit_row_ids"]
        == list(
            next(
                context.fit_row_ids
                for context in contexts
                if context.scope_id == row["canonical_owner_scope_id"]
            )
        )
        for row in accounting["physical_records"]
    )
    assert all(
        row["scope_id"].endswith("_hierarchy_epoch_001")
        and row["replacement_scope_id"].endswith("_inner_005")
        and row["same_fit_row_content_proven"] is False
        and row["current_equivalence_proven"] is True
        and row["legacy_order_reusable_for_current_fit"] is False
        and row["superseded_output_retained_by_identity_only"] is True
        for row in accounting["superseded_duplicate_outputs"]
    )

    unauthenticated = plan_legacy_v4_preflight_migration(
        manifest_path=manifest_path,
        logical_contexts=contexts,
        authenticate_registered_payload_bytes=False,
    )
    assert unauthenticated["decision"] == "recompute_required"
    assert (
        "registered_payload_bytes_not_freshly_authenticated"
        in unauthenticated["recompute_reason_codes"]
    )


def test_scoped_embedding_cache_denies_peer_rows(tmp_path: Path) -> None:
    array_path = tmp_path / "embeddings.npy"
    with array_path.open("xb") as handle:
        np.save(handle, np.arange(24, dtype=np.float32).reshape(6, 4))
    row_path = tmp_path / "rows.parquet"
    pd.DataFrame({"row_id": [f"p{value}" for value in range(6)]}).to_parquet(
        row_path,
        index=False,
    )
    cache = SharedEmbeddingCache(
        embedding_path=array_path,
        row_ids_path=row_path,
    )
    view = cache.scoped_view(("p1", "p3"))
    assert view.shape == (2, 4)
    assert view.take().flags.writeable is False
    with pytest.raises(PermissionError, match="peer rows"):
        view.take(("p1", "p2"))


def test_resource_scheduler_supports_cpu_and_heterogeneous_auto_inventory() -> None:
    occupied = GPUResource(
        device="cuda:0",
        uuid="occupied",
        total_memory_bytes=24 * GIB,
        free_memory_bytes=20 * GIB,
        utilization_percent=5.0,
        external_processes=({"pid": 123, "used_memory_bytes": GIB},),
    )
    available = GPUResource(
        device="cuda:3",
        uuid="available",
        total_memory_bytes=48 * GIB,
        free_memory_bytes=42 * GIB,
        utilization_percent=0.0,
    )
    inventory = ResourceInventory(cpu_count=16, gpus=(occupied, available))
    automatic = plan_resources(
        policy="auto",
        cpu_budget=4,
        inventory=inventory,
        cpu_supported=True,
        resource_performance_safety=_resource_safety(),
    )
    assert automatic.devices == ("cuda:3",)
    cpu = plan_resources(
        policy="cpu",
        cpu_budget=2,
        inventory=inventory,
        cpu_supported=True,
        resource_performance_safety=_resource_safety(),
    )
    assert cpu.devices == ("cpu",)
    with pytest.raises(RuntimeError, match="no external process was killed"):
        plan_resources(
            policy=("cuda:0",),
            cpu_budget=2,
            inventory=inventory,
            cpu_supported=True,
            resource_performance_safety=_resource_safety(),
        )


def test_resource_scheduler_applies_explicit_operational_device_count() -> None:
    inventory = ResourceInventory(
        cpu_count=16,
        gpus=tuple(
            GPUResource(
                device=f"cuda:{index}",
                uuid=f"available-{index}",
                total_memory_bytes=48 * GIB,
                free_memory_bytes=42 * GIB,
                utilization_percent=0.0,
            )
            for index in range(3)
        ),
    )
    selected = plan_resources(
        policy="auto",
        cpu_budget=4,
        requested_device_count=2,
        inventory=inventory,
        cpu_supported=True,
        resource_performance_safety=_resource_safety(),
    )
    assert selected.devices == ("cuda:0", "cuda:1")

    with pytest.raises(ValueError, match="exactly one"):
        plan_resources(
            policy="cpu",
            cpu_budget=4,
            requested_device_count=2,
            inventory=inventory,
            cpu_supported=True,
            resource_performance_safety=_resource_safety(),
        )
