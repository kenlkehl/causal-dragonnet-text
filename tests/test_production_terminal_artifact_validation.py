import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import oci.inference.production_terminal_artifact_validation as terminal_module
from oci.inference.production_oracle_evaluation import (
    evaluate_frozen_predictions_posthoc,
)
from oci.inference.production_terminal_artifact_validation import (
    validate_real_stage1_handoff,
    validate_real_stage2_canary,
    validate_real_stage2_terminal_artifacts,
)
from oci.inference.production_text_preparation import stable_file_sha256
from oci.models.strict_causal_forest_runtime import (
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    StrictCausalForestRuntimeConfig,
)
from tests.test_portable_workflow_contracts import (
    _forest_operational,
    _forest_spec,
    _generation_policy,
    _post_extraction_policy,
)


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _wrapped(path: Path, *, schema: str, body) -> Path:
    return _write_json(
        path,
        {
            "schema_version": schema,
            "content_sha256": _sha(body),
            "body": body,
        },
    )


def _artifact(path: Path):
    digest, size = stable_file_sha256(path)
    return {
        "relative_path": path.name,
        "path": str(path),
        "sha256": digest,
        "size_bytes": size,
    }


def _record(phase: str, paths):
    return {
        "phase": phase,
        "artifacts": [_artifact(Path(path).resolve()) for path in paths],
    }


def _record_for_phase(records, phase: str):
    return next(record for record in records if record["phase"] == phase)


def _registered_named_path(records, phase: str, name: str) -> Path:
    record = _record_for_phase(records, phase)
    return next(Path(row["path"]) for row in record["artifacts"] if Path(row["path"]).name == name)


def _refresh_record(record):
    record["artifacts"] = [_artifact(Path(row["path"])) for row in record["artifacts"]]


def _rewrite_flat_content_hash(path: Path, value):
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    value["content_sha256"] = _sha(body)
    _write_json(path, value)


def _rewrite_wrapper_content_hash(path: Path, value):
    value["content_sha256"] = _sha(value["body"])
    _write_json(path, value)


def _wire_budget():
    return {
        "budget_version": "hierarchy_wire_budget_v1",
        "max_opaque_identifier_chars": 128,
        "max_generated_name_chars": 64,
        "max_description_chars": 128,
        "max_reason_chars": 128,
        "max_ambiguity_chars": 128,
        "max_free_text_chars": 128,
        "max_generated_list_items": 8,
        "max_feature_names_per_member": 4,
        "max_findings_per_atomic_review": 4,
        "max_pair_relation_peers_per_page": 7,
        "max_definition_fold_inputs": 8,
        "max_group_lookback_ids": 8,
        "max_adaptive_review_targets": 4,
        "max_interpret_atoms_per_job": 2,
        "max_interpret_members_per_job": 3,
        "max_interpret_name_chars": 64,
        "max_interpret_description_chars": 96,
        "max_interpret_ambiguity_chars": 96,
        "max_interpret_reason_chars": 64,
        "max_interpret_canonical_json_bytes": 20_000,
        "max_interpret_transport_bytes": 20_000,
        "interpret_generation_token_budget": 20_000,
        "max_response_transport_bytes": 20_000,
        "generation_token_budget": 20_000,
    }


def _stage2_protocol():
    generation_policy = _generation_policy().as_dict()
    extraction_families = {
        "define_one_extraction_feature",
        "patient_feature_extraction",
    }
    for family, parameters in generation_policy.items():
        if family == "schema_version":
            continue
        parameters["max_tokens"] = 333 if family in extraction_families else 21_000
        parameters["thinking_token_budget"] = 111 if parameters["thinking_enabled"] else 0
    return {
        "proposal_max_tokens": 21_000,
        "extraction_max_tokens": 333,
        "model_context_window_tokens": 131_072,
        "post_extraction_review_max_operations": 4,
        "post_extraction_review_max_quality_retries": 2,
        "post_extraction_review_min_partition_rows": 3,
        "hierarchical_max_atoms_per_chunk": 2,
        "hierarchical_max_bytes_per_chunk": 1_024,
        "hierarchical_max_semantic_member_ids_per_chunk": 3,
        "hierarchical_max_cross_architecture_lookback_ids": 5,
        "hierarchical_max_cross_architecture_lookback_bytes": 2_048,
        "hierarchical_max_extraction_lookback_ids_per_feature": 4,
        "hierarchical_max_extraction_lookback_bytes_per_feature": 2_048,
        "hierarchical_max_rejection_lookback_ids_per_candidate": 6,
        "hierarchical_max_rejection_lookback_bytes_per_candidate": 2_048,
        "hierarchical_review_max_evidence_ids": 7,
        "hierarchical_review_max_evidence_bytes": 4_096,
        "max_rendered_discovery_prompt_bytes": 8_192,
        "selector_thinking_token_budget": 111,
        "final_upstream_max_orphan_features": 9,
        "review_neural_query_nuisance_folds": 3,
        "final_upstream_meta_inner_folds": 3,
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
        "hierarchy_wire_budget": _wire_budget(),
        "generation_policy": generation_policy,
    }


def _causal_review():
    return {
        "upstream_review_policy": "conditional_context_and_gate_v1",
        "e_clip": 0.05,
        "nuisance_ridge_alpha": 1.0,
        "effect_ridge_alpha": 1.0,
        "contract_complexity_penalty": 0.002,
        "encoded_column_complexity_penalty": 0.0002,
        "minimum_score_improvement": 0.0,
        "nuisance_relative_tolerance": 0.05,
        "source_preservation_tolerance": 0.05,
        "source_context_r_loss_relative_tolerance": 0.05,
        "feature_bank_preservation_tolerance": 0.05,
        "scientific_policy": _post_extraction_policy().as_dict(),
    }


def _tokenizer_tree(tmp_path: Path):
    root = (tmp_path / "tokenizer").resolve()
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_file = root / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file.write_text(
            '{"fixture":"tokenizer","version":1}\n',
            encoding="utf-8",
        )
    digest, size = stable_file_sha256(tokenizer_file)
    files = [
        {
            "relative_path": "tokenizer.json",
            "sha256": digest,
            "size_bytes": size,
        }
    ]
    return {
        "kind": "directory",
        "path": str(root),
        "file_count": 1,
        "total_size_bytes": size,
        "tree_sha256": _sha(files),
        "files": files,
    }


def _prompt_guard_identity(tmp_path: Path):
    tokenizer = _tokenizer_tree(tmp_path)
    body = {
        "schema_version": "stage2_prompt_nontruncation_v1",
        "model_name": "configured/model",
        "model_context_window_tokens": 131_072,
        "tokenizer_content_identity": {
            key: value for key, value in tokenizer.items() if key != "path"
        },
        "chat_template_sha256": "a" * 64,
        "tokenizer_class": {
            "module": "fixture.tokenizer",
            "qualname": "FixtureTokenizer",
        },
        "accounting": {
            "apply_chat_template": True,
            "tokenize": True,
            "add_generation_prompt": True,
            "truncation": False,
            "endpoint_prompt_usage_exact_match_required": True,
            "request_truncation_controls_allowed": False,
        },
    }
    return {**body, "identity_sha256": _sha(body)}


def _prompt_audit(
    *,
    guard_sha: str,
    request_sha: str,
    generation_tokens: int,
    local_prompt_tokens: int = 101,
    client_path: str = "hierarchical_discovery",
):
    context = 131_072
    required = local_prompt_tokens + generation_tokens
    body = {
        "schema_version": "stage2_prompt_nontruncation_v1",
        "guard_identity_sha256": guard_sha,
        "request_sha256": request_sha,
        "client_path": client_path,
        "local_prompt_tokens": local_prompt_tokens,
        "maximum_generation_tokens": generation_tokens,
        "required_context_tokens": required,
        "model_context_window_tokens": context,
        "context_headroom_tokens": context - required,
        "truncation_controls_present": False,
        "tokenizer_truncation_enabled": False,
        "endpoint_prompt_tokens": local_prompt_tokens,
        "endpoint_prompt_tokens_exact_match": True,
        "status": "accepted_nontruncated",
    }
    return {**body, "audit_sha256": _sha(body)}


def _prompt_execution_audit(*, guard_sha: str, records):
    counts = {
        "explicit_feature_extraction": 0,
        "hierarchical_discovery": 0,
        "proposal_and_post_extraction_review": 0,
    }
    for record in records:
        counts[record["client_path"]] += 1
    body = {
        "schema_version": ("stage2_prompt_nontruncation_execution_audit_v1"),
        "guard_identity_sha256": guard_sha,
        "record_count": len(records),
        "records": records,
        "records_sha256": _sha(records),
        "record_counts_by_client_path": counts,
        "unclassified_record_count": 0,
        "all_records_status": "accepted_nontruncated",
        "all_endpoint_prompt_tokens_exact_match": True,
        "all_request_audits_authenticated": True,
        "all_guard_identities_exact_match": True,
        "all_requests_forbid_truncation_controls": True,
    }
    return {**body, "audit_sha256": _sha(body)}


def _request(tmp_path: Path, *, oracle=False):
    return {
        "endpoint": "https://configured.example/v1",
        "model_name": "configured/model",
        "review_rounds": 2,
        "initial_training_partitions": 3,
        "interaction_inner_folds": 3,
        "tfidf_nested_calibration_folds": 3,
        "outer_folds": 2,
        "forest_n_estimators": 40,
        "forest_min_samples_leaf": 4,
        "forest_max_features": "sqrt",
        "forest_random_seed": 19,
        "evaluate_oracle_posthoc": oracle,
        "unit_id_column": "id",
        "oracle_ite_column": "true_ite_prob",
        "stage2_prompt_protocol": _stage2_protocol(),
        "post_extraction_causal_review": _causal_review(),
        "stage2_tokenizer_tree": _tokenizer_tree(tmp_path),
    }


def _handoff_fixture(tmp_path: Path):
    bundle_body = {
        "schema_version": "fixture_stage1_bundle",
        "request_sha256": "1" * 64,
    }
    bundle_sha = _sha(bundle_body)
    bundle = _write_json(
        tmp_path / "stage1" / "bundle_manifest.json",
        {**bundle_body, "bundle_sha256": bundle_sha},
    )
    handoff_body = {
        "schema_version": "fixture_handoff",
        "stage1_inputs": {"bundle_sha256": bundle_sha},
        "all_ten_architectures_required": True,
        "per_architecture_interpretation_required": True,
        "raw_all_architecture_prompt_allowed": False,
        "independent_runtime_stage1_refit_allowed": False,
        "manual_digest_approval_required": False,
    }
    handoff = {**handoff_body, "content_sha256": _sha(handoff_body)}
    report_body = {
        "schema_version": "production_stage1_fresh_handoff_validation_v1",
        "status": "accepted",
        "bundle_manifest_path": str(bundle),
        "review_rounds": 2,
        "initial_training_partitions": 3,
        "interaction_inner_folds": 3,
        "tfidf_nested_calibration_folds": 3,
        "handoff": handoff,
        "remote_clients_constructed": False,
        "remote_calls_made": False,
        "loader_module_path": str(
            Path("oci/inference/production_stage1_hierarchy_handoff.py").resolve()
        ),
    }
    report = _write_json(
        tmp_path / "handoff" / "fresh_handoff_validation.json",
        {**report_body, "content_sha256": _sha(report_body)},
    )
    return bundle, report, bundle_sha, handoff["content_sha256"]


def _canary_fixture(
    tmp_path: Path,
    *,
    bundle: Path,
    bundle_sha: str,
    handoff_sha: str,
    max_tokens: int = 21_000,
):
    guard = _prompt_guard_identity(tmp_path)
    request_sha = "b" * 64
    prompt_audit = _prompt_audit(
        guard_sha=guard["identity_sha256"],
        request_sha=request_sha,
        generation_tokens=max_tokens,
    )
    transport = {
        "job_id": "fixture",
        "job_kind": "interpret_evidence_chunk",
        "request_sha256": request_sha,
        "runner_identity_sha256": "2" * 64,
        "outcome": "success",
        "parsed_response_sha256": "c" * 64,
        "attempts": [
            {
                "attempt_number": 1,
                "endpoint": "https://configured.example/v1",
                "model": "configured/model",
                "request_sha256": request_sha,
                "runner_identity_sha256": "2" * 64,
                "response_model": "configured/model",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": prompt_audit["endpoint_prompt_tokens"],
                    "completion_tokens": 1,
                    "total_tokens": (prompt_audit["endpoint_prompt_tokens"] + 1),
                },
                "prompt_nontruncation_audit": prompt_audit,
                "outcome": "success",
                "retryable": False,
                "will_retry": False,
            }
        ],
    }
    protocol = {
        "schema_version": "stage2_hierarchy_prompt_protocol_v3",
        **_stage2_protocol(),
    }
    protocol["proposal_max_tokens"] = max_tokens
    body = {
        "status": "accepted",
        "canary_kind": "one_real_architecture_pure_initial_interpretation_job",
        "authorization_role": "non_authorizing_operational_runtime_check",
        "stage1_bundle": {
            "manifest_path": str(bundle),
            "bundle_sha256": bundle_sha,
            "handoff_content_sha256": handoff_sha,
        },
        "endpoint": "https://configured.example/v1",
        "model": "configured/model",
        "runner_identity_sha256": "2" * 64,
        "settings": {
            "proposal_max_tokens": max_tokens,
            "extraction_max_tokens": protocol["extraction_max_tokens"],
            "stage2_hierarchy_prompt_protocol": protocol,
            "stage2_hierarchy_prompt_protocol_sha256": _sha(protocol),
            "post_extraction_causal_review": _causal_review(),
            "post_extraction_causal_review_sha256": _sha(_causal_review()),
            "prompt_nontruncation_guard_identity_sha256": guard["identity_sha256"],
            "transport_retries": 0,
            "selector_thinking_enabled": True,
            "selector_thinking_token_budget": protocol["selector_thinking_token_budget"],
            "max_rendered_discovery_prompt_bytes": protocol["max_rendered_discovery_prompt_bytes"],
            "final_upstream_max_orphan_features": protocol["final_upstream_max_orphan_features"],
            "review_neural_query_nuisance_folds": protocol["review_neural_query_nuisance_folds"],
            "final_upstream_meta_inner_folds": protocol["final_upstream_meta_inner_folds"],
            "final_upstream_head_regularization": protocol["final_upstream_head_regularization"],
            "extraction_thinking_enabled": False,
            "maximum_schema_repairs": 1,
        },
        "selected_job": {"job_id": "fixture"},
        "validation": {
            "normalized_response_sha256": "3" * 64,
            "raw_wire_response_sha256": "4" * 64,
            "response_attempt_trace_sha256": "5" * 64,
            "response_attempt_outcomes": ["validated_response"],
            "local_json_schema_validator_identity_sha256": "6" * 64,
            "response_repair_policy_sha256": "7" * 64,
            "job_cache_identity_sha256": "8" * 64,
            "validated_only_cache_enabled": True,
        },
        "remote_response_count": 1,
        "transport_metadata": [transport],
        "raw_prompt_emitted": False,
        "raw_response_emitted": False,
        "normalized_findings_emitted": False,
        "prediction_path_constructed": False,
        "oracle_path_constructed": False,
        "full_fusion_runner_executed": False,
        "canary_implementation_file_sha256": "9" * 64,
    }
    return _wrapped(
        tmp_path / "canary" / "production_stage1_hierarchy_runtime_canary.json",
        schema="production_stage1_hierarchy_runtime_canary_report_v2",
        body=body,
    )


def test_fresh_handoff_and_canary_are_deeply_validated_without_fixed_token_budget(
    tmp_path,
):
    bundle, report, bundle_sha, handoff_sha = _handoff_fixture(tmp_path)
    canary = _canary_fixture(
        tmp_path,
        bundle=bundle,
        bundle_sha=bundle_sha,
        handoff_sha=handoff_sha,
        max_tokens=21_000,
    )
    records = [
        _record("stage1_modeling", [bundle]),
        _record("handoff_validation", [report]),
        _record("stage2_canary", [canary]),
    ]
    handoff = validate_real_stage1_handoff(
        request=_request(tmp_path),
        phase_records=records,
    )
    result = validate_real_stage2_canary(
        request=_request(tmp_path),
        phase_records=records,
        handoff_validation=handoff,
    )
    assert result["finish_reason_stop_proven"] is True
    assert result["transport_retries"] == 0
    assert result["prompt_nontruncation_execution_audits_validated"] == 1


def test_fresh_handoff_and_canary_fail_on_partition_or_response_metadata_change(
    tmp_path,
):
    bundle, report, bundle_sha, handoff_sha = _handoff_fixture(tmp_path)
    canary = _canary_fixture(
        tmp_path,
        bundle=bundle,
        bundle_sha=bundle_sha,
        handoff_sha=handoff_sha,
    )
    report_value = json.loads(report.read_text(encoding="utf-8"))
    report_value["initial_training_partitions"] = 4
    report_body = {key: value for key, value in report_value.items() if key != "content_sha256"}
    report_value["content_sha256"] = _sha(report_body)
    _write_json(report, report_value)
    records = [
        _record("stage1_modeling", [bundle]),
        _record("handoff_validation", [report]),
        _record("stage2_canary", [canary]),
    ]
    with pytest.raises(ValueError, match="handoff validation report"):
        validate_real_stage1_handoff(
            request=_request(tmp_path),
            phase_records=records,
        )

    _bundle, report, _bundle_sha, _handoff_sha = _handoff_fixture(tmp_path)
    wrapper = json.loads(canary.read_text(encoding="utf-8"))
    wrapper["body"]["transport_metadata"][0]["attempts"][0]["finish_reason"] = "length"
    wrapper["content_sha256"] = _sha(wrapper["body"])
    _write_json(canary, wrapper)
    records = [
        _record("stage1_modeling", [bundle]),
        _record("handoff_validation", [report]),
        _record("stage2_canary", [canary]),
    ]
    handoff = validate_real_stage1_handoff(
        request=_request(tmp_path),
        phase_records=records,
    )
    with pytest.raises(ValueError, match="response metadata"):
        validate_real_stage2_canary(
            request=_request(tmp_path),
            phase_records=records,
            handoff_validation=handoff,
        )


def _terminal_fixture(tmp_path: Path, *, oracle: bool):
    bundle, report, bundle_sha, handoff_sha = _handoff_fixture(tmp_path)
    row_map = tmp_path / "stage1" / "row_registry.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2, 3],
            "id": ["p0", "p1", "p2", "p3"],
        }
    ).to_parquet(row_map, index=False)
    row_map = row_map.resolve()
    canary = _canary_fixture(
        tmp_path,
        bundle=bundle,
        bundle_sha=bundle_sha,
        handoff_sha=handoff_sha,
    )
    combined = pd.DataFrame(
        {
            "_oci_row_id": [0, 1, 2, 3],
            "outer_fold": [1, 2, 1, 2],
            "pred_y0_prob": [0.2, 0.3, 0.4, 0.5],
            "pred_y1_prob": [0.4, 0.2, 0.7, 0.6],
        }
    )
    combined["pred_ite_prob"] = combined["pred_y1_prob"] - combined["pred_y0_prob"]
    inference_root = tmp_path / "inference"
    combined_path = inference_root / "frozen_predictions.parquet"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path, index=False)
    combined_path = combined_path.resolve()
    inference_paths = [combined_path]
    fold_manifests = []
    for fold in (1, 2):
        fold_dir = inference_root / f"fold_{fold:03d}"
        fold_dir.mkdir()
        frame = combined[combined["outer_fold"] == fold].reset_index(drop=True)
        prediction_path = fold_dir / "frozen_predictions.parquet"
        frame.to_parquet(prediction_path, index=False)
        prediction_path = prediction_path.resolve()
        prediction_sha, _size = stable_file_sha256(prediction_path)
        estimator = {
            "mode": "strict_outer_honest_final_context_fit_causal_forest_v2",
            "strict_causal_forest_active": True,
            "strict_causal_forest_required": True,
            "structured_interaction_head_constructed": False,
            "outer_heldout_labels_used": False,
            "forest_backend_identity": {
                "identity": {
                    "n_estimators": 40,
                    "min_samples_leaf": 4,
                    "max_features": "sqrt",
                    "honest": True,
                    "inference": True,
                    "random_state": 19,
                }
            },
        }
        from oci.inference.fold_honest_signal_fusion import row_set_fingerprint

        body = {
            "outer_fold": fold,
            "train_row_count": 2,
            "heldout_row_count": 2,
            "heldout_row_fingerprint": row_set_fingerprint(frame["_oci_row_id"].tolist()),
            "outer_heldout_outcomes_used": False,
            "oracle_columns_written": False,
            "prediction_columns": list(frame.columns),
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "final_ite_estimator": estimator,
        }
        manifest = _wrapped(
            fold_dir / "immutable_fold_manifest.json",
            schema="all_evidence_fusion_frozen_fold_v20",
            body=body,
        )
        fold_manifests.append(manifest)
        inference_paths.extend((prediction_path, manifest))
    combined_sha, _size = stable_file_sha256(combined_path)
    run_body = {
        "fold_manifest_paths": [str(path) for path in fold_manifests],
        "fold_count": 2,
        "prediction_path": str(combined_path),
        "prediction_sha256": combined_sha,
        "prediction_row_count": 4,
        "prediction_columns": list(combined.columns),
        "outer_test_rows_predicted_once": True,
        "final_ite_estimator": {
            "mode": "strict_outer_honest_final_context_fit_causal_forest_v2",
            "strict_causal_forest_active_for_every_fold": True,
            "strict_causal_forest_required": True,
            "fixed_prior_working_backend_active": True,
        },
        "oracle_columns_written": False,
    }
    run_manifest = _wrapped(
        inference_root / "immutable_run_manifest.json",
        schema="all_evidence_fusion_predictions_v5",
        body=run_body,
    )
    inference_paths.append(run_manifest)
    batch = _wrapped(
        inference_root / "preparation" / "authenticated_hierarchical_batch_result.json",
        schema="hierarchical_all_evidence_runner_batch_result_v1",
        body={
            "batch_result_sha256": "d" * 64,
            "all_fold_discovery_completed_before_per_fold_modeling": True,
        },
    )
    inference_paths.append(batch)
    guard = _prompt_guard_identity(tmp_path)
    run_sha, _run_size = stable_file_sha256(run_manifest)
    batch_sha, _batch_size = stable_file_sha256(batch)
    one_shot_fold_rows = []
    for path in fold_manifests:
        digest, size = stable_file_sha256(path)
        one_shot_fold_rows.append({"path": str(path), "size": size, "sha256": digest})
    one_shot_body = {
        "schema_version": ("production_stage1_hierarchy_one_shot_attestation_v2"),
        "status": "completed",
        "stage1_bundle_manifest_path": str(bundle),
        "stage1_bundle_sha256": bundle_sha,
        "stage1_handoff_content_sha256": handoff_sha,
        "stage1_provider_identity_sha256": "e" * 64,
        "production_endpoint": "https://configured.example/v1",
        "production_model": "configured/model",
        "stage2_hierarchy_prompt_protocol": {
            "schema_version": "stage2_hierarchy_prompt_protocol_v3",
            **_stage2_protocol(),
        },
        "stage2_hierarchy_prompt_protocol_sha256": _sha(
            {
                "schema_version": "stage2_hierarchy_prompt_protocol_v3",
                **_stage2_protocol(),
            }
        ),
        "post_extraction_causal_review": _causal_review(),
        "post_extraction_causal_review_sha256": _sha(_causal_review()),
        "remote_runtime_identity": {
            "endpoint_urls": ["https://configured.example/v1"],
            "model": {"name": "configured/model"},
            "guarded_client_paths": [
                "hierarchical_discovery",
                "proposal_and_post_extraction_review",
                "explicit_feature_extraction",
            ],
            "endpoint_pool_or_fallback_allowed": False,
            "model_autodiscovery_or_substitution_allowed": False,
            "required_response_model": "configured/model",
            "required_finish_reason": "stop",
            "response_metadata_checked_before_content_semantics_and_cache": True,
            "prompt_nontruncation_guard": guard,
            "local_prompt_tokens_plus_generation_within_context_required": True,
            "endpoint_prompt_token_usage_exact_match_required": True,
            "request_prompt_truncation_controls_allowed": False,
            "served_deployment_metadata_required": False,
            "caller_digest_authority": False,
        },
        "prompt_nontruncation_execution_audit": (
            _prompt_execution_audit(
                guard_sha=guard["identity_sha256"],
                records=[
                    _prompt_audit(
                        guard_sha=guard["identity_sha256"],
                        request_sha="7" * 64,
                        generation_tokens=_stage2_protocol()["proposal_max_tokens"],
                    )
                ],
            )
        ),
        "hierarchical_runner_identity_sha256": "f" * 64,
        "preparation_dir": str(batch.parent),
        "hierarchical_batch_result": {
            "path": str(batch),
            "sha256": batch_sha,
        },
        "final_output_dir": str(inference_root.resolve()),
        "immutable_run_manifest": {
            "path": str(run_manifest),
            "sha256": run_sha,
            "content_sha256": json.loads(run_manifest.read_text(encoding="utf-8"))[
                "content_sha256"
            ],
        },
        "frozen_predictions": {
            "path": str(combined_path),
            "size": stable_file_sha256(combined_path)[1],
            "sha256": combined_sha,
        },
        "fold_manifests": one_shot_fold_rows,
        "one_shot_implementation_sha256": "1" * 64,
        "run_result_audit_record_is_authorization": False,
        "architecture_at_a_time_hierarchy_required": True,
        "same_handoff_provider_used_for_spent_and_partitions": True,
        "genuine_one_shot_e2e_certified": False,
        "global_certification_mutated": False,
    }
    one_shot = _write_json(
        inference_root / "attestation" / "production_stage1_hierarchy_one_shot_result.json",
        {
            **one_shot_body,
            "content_sha256": _sha(one_shot_body),
        },
    )
    inference_paths.append(one_shot)
    records = [
        _record("stage1_modeling", [bundle, row_map]),
        _record("handoff_validation", [report]),
        _record("stage2_canary", [canary]),
        _record("stage2_inference", inference_paths),
    ]
    if oracle:
        oracle_path = tmp_path / "oracle.parquet"
        pd.DataFrame(
            {
                "id": ["p0", "p1", "p2", "p3"],
                "true_ite_prob": [0.1, -0.2, 0.25, 0.05],
            }
        ).to_parquet(oracle_path, index=False)
        evaluation = evaluate_frozen_predictions_posthoc(
            predictions_path=combined_path,
            prediction_manifest_path=run_manifest,
            unit_id_map_path=row_map,
            oracle_dataset_path=oracle_path,
            output_dir=tmp_path / "evaluation",
            unit_id_column="id",
            oracle_unit_id_column="id",
            oracle_ite_column="true_ite_prob",
        )
        records.append(
            _record(
                "oracle_evaluation",
                [
                    Path(evaluation["joined_path"]),
                    tmp_path / "evaluation" / "evaluation_metrics.json",
                ],
            )
        )
    else:
        records.append(_record("oracle_evaluation", []))
    return records, row_map


def test_terminal_validator_reopens_fold_predictions_and_exact_row_order(
    tmp_path,
):
    records, row_map = _terminal_fixture(tmp_path, oracle=False)
    result = validate_real_stage2_terminal_artifacts(
        request=_request(tmp_path),
        phase_records=records,
    )
    assert result["prediction_row_count"] == 4
    assert result["fold_prediction_count"] == 2
    assert result["row_order_validated"] is True
    assert result["probability_scale_identity_validated"] is True
    assert (
        result["stage2_one_shot_validation"]["prompt_nontruncation_guard_identity_sha256"]
        == result["stage2_canary_validation"]["prompt_nontruncation_guard_identity_sha256"]
    )

    changed = pd.read_parquet(row_map).iloc[::-1].reset_index(drop=True)
    changed.to_parquet(row_map, index=False)
    stage1 = next(record for record in records if record["phase"] == "stage1_modeling")
    stage1["artifacts"] = [_artifact(Path(row["path"])) for row in stage1["artifacts"]]
    with pytest.raises(ValueError, match="row map order"):
        validate_real_stage2_terminal_artifacts(
            request=_request(tmp_path),
            phase_records=records,
        )


@pytest.mark.parametrize(
    ("tampering", "message"),
    [
        ("missing_attestation", "exactly one"),
        ("causal_review", "scientific prompt/review"),
        ("wire_budget", "scientific prompt/review"),
        ("guard_accounting", "prompt-guard identity"),
        ("missing_execution_audit", "one-shot attestation"),
        ("execution_summary", "execution audit"),
        ("execution_record", "execution record"),
        ("canary_usage_audit", "prompt nontruncation audit"),
        ("missing_canary_audit", "prompt nontruncation audit"),
        ("tokenizer_bytes", "tokenizer bytes"),
    ],
)
def test_terminal_validator_fails_closed_on_stage2_scientific_and_prompt_proof_tampering(
    tmp_path,
    tampering,
    message,
):
    records, _row_map = _terminal_fixture(tmp_path, oracle=False)
    immutable_request = _request(tmp_path)
    inference = _record_for_phase(records, "stage2_inference")
    canary_record = _record_for_phase(records, "stage2_canary")
    attestation_path = _registered_named_path(
        records,
        "stage2_inference",
        "production_stage1_hierarchy_one_shot_result.json",
    )
    canary_path = _registered_named_path(
        records,
        "stage2_canary",
        "production_stage1_hierarchy_runtime_canary.json",
    )

    if tampering == "missing_attestation":
        inference["artifacts"] = [
            row
            for row in inference["artifacts"]
            if Path(row["path"]).name != "production_stage1_hierarchy_one_shot_result.json"
        ]
    elif tampering in {
        "causal_review",
        "wire_budget",
        "guard_accounting",
        "missing_execution_audit",
        "execution_summary",
        "execution_record",
    }:
        value = json.loads(attestation_path.read_text(encoding="utf-8"))
        if tampering == "causal_review":
            value["post_extraction_causal_review"]["e_clip"] = 0.06
            value["post_extraction_causal_review_sha256"] = _sha(
                value["post_extraction_causal_review"]
            )
        elif tampering == "wire_budget":
            protocol = value["stage2_hierarchy_prompt_protocol"]
            protocol["hierarchy_wire_budget"]["max_free_text_chars"] += 1
            value["stage2_hierarchy_prompt_protocol_sha256"] = _sha(protocol)
        elif tampering == "guard_accounting":
            guard = value["remote_runtime_identity"]["prompt_nontruncation_guard"]
            guard["accounting"]["truncation"] = True
            guard_body = {key: child for key, child in guard.items() if key != "identity_sha256"}
            guard["identity_sha256"] = _sha(guard_body)
        elif tampering == "missing_execution_audit":
            value.pop("prompt_nontruncation_execution_audit")
        elif tampering == "execution_summary":
            execution = value["prompt_nontruncation_execution_audit"]
            execution["unclassified_record_count"] = 1
            execution_body = {
                key: child for key, child in execution.items() if key != "audit_sha256"
            }
            execution["audit_sha256"] = _sha(execution_body)
        else:
            execution = value["prompt_nontruncation_execution_audit"]
            record = execution["records"][0]
            record["truncation_controls_present"] = True
            record_body = {key: child for key, child in record.items() if key != "audit_sha256"}
            record["audit_sha256"] = _sha(record_body)
            execution["records_sha256"] = _sha(execution["records"])
            execution_body = {
                key: child for key, child in execution.items() if key != "audit_sha256"
            }
            execution["audit_sha256"] = _sha(execution_body)
        _rewrite_flat_content_hash(attestation_path, value)
        _refresh_record(inference)
    elif tampering in {"canary_usage_audit", "missing_canary_audit"}:
        wrapper = json.loads(canary_path.read_text(encoding="utf-8"))
        attempt = wrapper["body"]["transport_metadata"][0]["attempts"][0]
        if tampering == "missing_canary_audit":
            attempt.pop("prompt_nontruncation_audit")
        else:
            audit = attempt["prompt_nontruncation_audit"]
            audit["endpoint_prompt_tokens"] += 1
            audit_body = {key: child for key, child in audit.items() if key != "audit_sha256"}
            audit["audit_sha256"] = _sha(audit_body)
            attempt["usage"]["prompt_tokens"] = audit["endpoint_prompt_tokens"]
        _rewrite_wrapper_content_hash(canary_path, wrapper)
        _refresh_record(canary_record)
    else:
        tokenizer_file = Path(immutable_request["stage2_tokenizer_tree"]["path"]) / "tokenizer.json"
        tokenizer_file.write_text(
            '{"fixture":"mutated-tokenizer","version":2}\n',
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match=message):
        validate_real_stage2_terminal_artifacts(
            request=immutable_request,
            phase_records=records,
        )


def test_terminal_validator_recomputes_metrics_and_proves_exact_oracle_events(
    tmp_path,
):
    records, _row_map = _terminal_fixture(tmp_path, oracle=True)
    result = validate_real_stage2_terminal_artifacts(
        request=_request(tmp_path, oracle=True),
        phase_records=records,
    )
    assert result["oracle_validation"]["oracle_open_order_proven"] is True

    evaluation = next(record for record in records if record["phase"] == "oracle_evaluation")
    metrics_path = next(
        Path(row["path"])
        for row in evaluation["artifacts"]
        if Path(row["path"]).name == "evaluation_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["event_order"][3]["event"] = "oracle_source_opened"
    _write_json(metrics_path, metrics)
    evaluation["artifacts"] = [_artifact(Path(row["path"])) for row in evaluation["artifacts"]]
    with pytest.raises(ValueError, match="event ordering"):
        validate_real_stage2_terminal_artifacts(
            request=_request(tmp_path, oracle=True),
            phase_records=records,
        )


def _portable_forest_request(tmp_path: Path):
    runtime = StrictCausalForestRuntimeConfig(
        schema_version=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
        causal_forest=_forest_spec(),
        operational=_forest_operational(6),
    )
    return {
        **_request(tmp_path),
        "forest_runtime_config": runtime.as_dict(),
        "forest_n_estimators": None,
        "forest_max_depth": None,
        "forest_min_samples_leaf": None,
        "forest_max_features": None,
        "forest_honest": None,
        "forest_inference": None,
        "forest_subforest_size": None,
        "forest_tune_model": None,
        "forest_nuisance_n_estimators": None,
        "forest_nuisance_max_depth": None,
        "forest_nuisance_min_samples_leaf": None,
        "forest_nuisance_treatment_max_features": None,
        "forest_nuisance_outcome_max_features": None,
        "forest_random_seed": None,
        "forest_n_jobs": None,
        "cpu_budget": 6,
    }


def _portable_forest_fold(
    tmp_path: Path,
    *,
    runtime_override=None,
):
    request = _portable_forest_request(tmp_path)
    strict_runtime = StrictCausalForestRuntimeConfig.from_mapping(request["forest_runtime_config"])
    runtime = (
        {
            "causal_forest_head_module_sha256": "a" * 64,
            "econml_distribution_version": "fixture",
        }
        if runtime_override is None
        else dict(runtime_override)
    )
    scientific = strict_runtime.causal_forest
    configured = {
        "n_estimators": scientific.n_estimators,
        "max_depth": scientific.max_depth,
        "min_samples_leaf": scientific.min_samples_leaf,
        "max_features": scientific.max_features,
        "honest": scientific.honest,
        "inference": scientific.inference,
        "subforest_size": scientific.subforest_size,
        "random_state": scientific.random_seed,
    }
    treatment = strict_runtime.treatment_constructor_kwargs()
    outcome = strict_runtime.outcome_constructor_kwargs()
    crossfit = strict_runtime.crossfit_constructor_kwargs()
    top_level = {
        **scientific.scientific_constructor_kwargs(),
        "n_jobs": 1,
        "verbose": strict_runtime.operational.verbose,
        "use_ray": strict_runtime.operational.use_ray,
        "ray_remote_func_options": (strict_runtime.operational.ray_remote_func_options),
    }
    unfitted = {
        "top_level_attributes": top_level,
        "model_t_parameters": treatment,
        "model_y_parameters": outcome,
        "crossfit_parameters": crossfit,
    }
    grf = {
        "criterion": scientific.criterion,
        "fit_intercept": scientific.fit_intercept,
        "honest": scientific.honest,
        "inference": scientific.inference,
        "max_depth": scientific.max_depth,
        "max_features": scientific.max_features,
        "max_samples": scientific.max_samples,
        "min_balancedness_tol": scientific.min_balancedness_tol,
        "min_impurity_decrease": scientific.min_impurity_decrease,
        "min_samples_leaf": scientific.min_samples_leaf,
        "min_samples_split": scientific.min_samples_split,
        "min_var_fraction_leaf": scientific.min_var_fraction_leaf,
        "min_var_leaf_on_val": scientific.min_var_leaf_on_val,
        "min_weight_fraction_leaf": scientific.min_weight_fraction_leaf,
        "n_estimators": scientific.n_estimators,
        "n_jobs": 1,
        "random_state": scientific.random_seed,
        "subforest_size": scientific.subforest_size,
        "verbose": strict_runtime.operational.verbose,
        "warm_start": False,
    }
    fitted = {
        "unfitted_estimator_graph": unfitted,
        "fitted_treatment_models": [[treatment for _fold in range(scientific.crossfit.n_splits)]],
        "fitted_outcome_models": [[outcome for _fold in range(scientific.crossfit.n_splits)]],
        "model_cate_template_parameters": grf,
        "fitted_grf_parameters": [grf],
    }
    split_body = {
        "implementation": scientific.crossfit.implementation,
        "parameters": crossfit,
        "splits": [
            {
                "fold_index": fold,
                "train_count": 4,
                "test_count": 4,
                "train_index_sha256": f"{fold + 1:x}" * 64,
                "test_index_sha256": f"{fold + 3:x}" * 64,
            }
            for fold in range(scientific.crossfit.n_splits)
        ],
    }
    split_audit = {
        **split_body,
        "split_plan_sha256": _sha(split_body),
    }
    identity = {
        "backend": "repository_strict_causal_forest_path_v4",
        "configuration_mode": "portable_strict_runtime_config_v1",
        "strict_runtime_scientific_identity": (strict_runtime.scientific_identity()),
        "strict_runtime_scientific_identity_sha256": (strict_runtime.scientific_identity_sha256()),
        "operational_settings_excluded_from_scientific_identity": True,
        "exact_nuisance_used_as_fixed_internal_predictions": False,
        "tuning_labels": "outer_train_only",
        "outer_heldout_labels_accepted": False,
        "repository_runtime": runtime,
    }
    fit_call_contract = dict(strict_runtime.scientific_identity()["fit_contract"])
    fit_call_contract.pop("prediction_contrast")
    fit = {
        "configuration_mode": "portable_strict_runtime_config_v1",
        "runtime_schema_version": strict_runtime.schema_version,
        "scientific_identity": strict_runtime.scientific_identity(),
        "scientific_identity_sha256": (strict_runtime.scientific_identity_sha256()),
        "operational_attestation": (strict_runtime.operational_attestation()),
        "operational_parameters": (strict_runtime.operational_attestation()),
        "tuning_configured": False,
        "tuning_attempted": False,
        "tuning_succeeded": None,
        "tuning_failure_fell_back_to_configured_parameters": False,
        "tuning_params": None,
        "crossfit_split_audit": split_audit,
        "unfitted_estimator_audit": unfitted,
        "fitted_estimator_audit": fitted,
        "fit_call_contract": fit_call_contract,
        "prediction_contrast": {"T0": 0, "T1": 1},
        "effective_parameters": configured,
        "effective_nuisance_parameters": {
            "treatment_model": treatment,
            "outcome_model": outcome,
        },
        "outer_train_labels_only": True,
        "outer_heldout_labels_accepted": False,
        "repository_runtime": runtime,
    }
    receipt_body = {
        "schema_version": ("role_neutral_direct_strict_causal_forest_receipt_v1"),
        "outer_fold": 1,
        "backend_identity": {
            "identity": identity,
            "identity_sha256": _sha(identity),
        },
        "backend_fit_audit": fit,
        "reference_manifest_content_sha256": "6" * 64,
        "effect_train_sha256": "7" * 64,
        "effect_heldout_sha256": "8" * 64,
        "control_train_sha256": "9" * 64,
        "control_heldout_sha256": "a" * 64,
        "treatment_sha256": "b" * 64,
        "outcome_sha256": "c" * 64,
        "tau_sha256": "d" * 64,
        "probability_difference_bounds": [-1.0, 1.0],
        "probability_difference_validation_tolerance": float(64 * np.finfo(np.float64).eps),
        "probability_difference_bounds_validated": True,
        "probability_difference_values_clipped": False,
        "effect_column_count": 3,
        "control_column_count": 2,
        "explicit_effect_column_count": 1,
        "explicit_control_column_count": 1,
        "fit_row_count": 8,
        "prediction_row_count": 2,
        "strict_causal_forest_only": True,
        "structured_or_nonforest_fallback_used": False,
        "outer_heldout_labels_used": False,
        "potential_outcome_columns_emitted": False,
    }
    fold = {
        "outer_fold": 1,
        "train_row_count": 8,
        "heldout_row_count": 2,
        "final_ite_estimator": {
            "mode": ("strict_outer_honest_final_context_fit_causal_forest_v2"),
            "strict_causal_forest_active": True,
            "strict_causal_forest_required": True,
            "structured_interaction_head_constructed": False,
            "outer_heldout_labels_used": False,
            "reference_only_role_neutral_runtime": True,
            "potential_outcome_reconstruction": ("not_emitted_direct_cate_estimand_only"),
            "forest_backend_identity": {
                "identity": identity,
                "identity_sha256": _sha(identity),
            },
            "forest_receipt": {
                **receipt_body,
                "content_sha256": _sha(receipt_body),
            },
        },
    }
    return request, runtime, fold


def test_portable_direct_prediction_schema_is_cate_only_and_closed():
    direct = pd.DataFrame(
        {
            "_oci_row_id": [0, 1],
            "outer_fold": [1, 2],
            "pred_ite_prob": [-0.2, 0.3],
        }
    )
    terminal_module._validate_prediction_frame(
        direct,
        label="portable CATE",
        prediction_columns=terminal_module._DIRECT_CATE_PREDICTION_COLUMNS,
    )
    fabricated = direct.assign(
        pred_y0_prob=[0.4, 0.5],
        pred_y1_prob=[0.2, 0.8],
    )[
        [
            "_oci_row_id",
            "outer_fold",
            "pred_y0_prob",
            "pred_y1_prob",
            "pred_ite_prob",
        ]
    ]
    with pytest.raises(ValueError, match="closed prediction schema"):
        terminal_module._validate_prediction_frame(
            fabricated,
            label="portable CATE",
            prediction_columns=(terminal_module._DIRECT_CATE_PREDICTION_COLUMNS),
        )
    direct.loc[0, "pred_ite_prob"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        terminal_module._validate_prediction_frame(
            direct,
            label="portable CATE",
            prediction_columns=(terminal_module._DIRECT_CATE_PREDICTION_COLUMNS),
        )


def test_portable_strict_forest_reopens_all_scientific_and_fit_settings(
    tmp_path,
):
    request, runtime, fold = _portable_forest_fold(tmp_path)
    terminal_module._validate_strict_forest(
        fold_body=fold,
        request=request,
        portable_direct=True,
        freshly_authenticated_repository_runtime=runtime,
        expected_direct_numerical_manifest_sha256="6" * 64,
    )

    legacy_backend = json.loads(json.dumps(fold))
    legacy_identity = legacy_backend["final_ite_estimator"]["forest_backend_identity"]["identity"]
    legacy_identity["backend"] = "repository_strict_causal_forest_path_v3"
    legacy_backend["final_ite_estimator"]["forest_backend_identity"]["identity_sha256"] = _sha(
        legacy_identity
    )
    with pytest.raises(ValueError, match="strict v4"):
        terminal_module._validate_strict_forest(
            fold_body=legacy_backend,
            request=request,
            portable_direct=True,
            freshly_authenticated_repository_runtime=runtime,
            expected_direct_numerical_manifest_sha256="6" * 64,
        )

    changed = json.loads(json.dumps(fold))
    changed["final_ite_estimator"]["forest_receipt"]["backend_fit_audit"][
        "effective_nuisance_parameters"
    ]["treatment_model"]["min_samples_leaf"] += 1
    with pytest.raises(ValueError, match="fit audit"):
        terminal_module._validate_strict_forest(
            fold_body=changed,
            request=request,
            portable_direct=True,
            freshly_authenticated_repository_runtime=runtime,
            expected_direct_numerical_manifest_sha256="6" * 64,
        )

    tuned_request = json.loads(json.dumps(request))
    tuned_request["forest_runtime_config"]["causal_forest"]["tune_model"] = True
    with pytest.raises(ValueError, match="closed strict-forest"):
        terminal_module._validate_strict_forest(
            fold_body=fold,
            request=tuned_request,
            portable_direct=True,
            freshly_authenticated_repository_runtime=runtime,
            expected_direct_numerical_manifest_sha256="6" * 64,
        )


def test_portable_stage1_binding_requires_the_closed_reference_inventory(
    tmp_path,
):
    root = tmp_path.resolve()
    execution = _write_json(
        root / "execution" / "execution_manifest.json",
        {"schema_version": "fixture", "content_sha256": "3" * 64},
    )
    bundle_value = {
        "request_sha256": "2" * 64,
        "scope_plan_scientific_content_sha256": "4" * 64,
        "source_role_neutral_execution_content_sha256": "3" * 64,
        "physical_fit_count": 35,
        "logical_scope_count": 40,
        "bundle_sha256": "5" * 64,
    }
    bundle = _write_json(
        root / "reference" / "bundle_manifest.json",
        bundle_value,
    )
    numerical = _write_json(
        root
        / "direct_upstream_numerical_reference_bank"
        / "direct_upstream_numerical_manifest.json",
        {"schema_version": "fixture", "content_sha256": "6" * 64},
    )
    locator = _write_json(
        numerical.parent / "locator_attestation.json",
        {"schema_version": "fixture"},
    )
    execution_sha, execution_size = stable_file_sha256(execution)
    bundle_sha, bundle_size = stable_file_sha256(bundle)
    integration_body = {
        "schema_version": ("production_role_neutral_stage1_integration_code_identity_v1"),
        "producer_factories_builder": {"source_sha256": "7" * 64},
        "physical_owner_executor": {"source_sha256": "8" * 64},
        "stage2_handoff_publisher": {"source_sha256": "9" * 64},
    }
    integration = {
        **integration_body,
        "content_sha256": _sha(integration_body),
    }
    binding_body = {
        "schema_version": ("production_portable_role_neutral_stage1_handoff_binding_v1"),
        "workflow_request_sha256": "1" * 64,
        "prepared_stage1_request_sha256": "2" * 64,
        "stage1_scope_plan_scientific_content_sha256": "4" * 64,
        "role_neutral_execution_manifest": {
            "relative_path": execution.relative_to(root).as_posix(),
            "sha256": execution_sha,
            "size_bytes": execution_size,
            "content_sha256": "3" * 64,
        },
        "stage2_bundle_manifest": {
            "relative_path": bundle.relative_to(root).as_posix(),
            "sha256": bundle_sha,
            "size_bytes": bundle_size,
            "bundle_sha256": "5" * 64,
        },
        "direct_numerical_reference_bank": {
            "relative_path": numerical.relative_to(root).as_posix(),
            "content_sha256": "6" * 64,
            "source_execution_content_sha256": "3" * 64,
            "combined_npy_payloads_persisted": False,
        },
        "integration_code_identity": integration,
        "physical_fit_count": 35,
        "logical_scope_count": 40,
        "deduplicated_fit_count": 5,
        "productive_compute_canary_completed": False,
        "selected_canary_replica_adopted_as_production": False,
        "compute_canary_scientific_equality": None,
        "legacy_bundle_build_invoked": False,
        "all_ten_role_neutral_execution_is_exclusive_evidence_source": True,
        "stage2_loader_validation": ("reference_only_role_neutral_provider_accepted"),
    }
    binding = _write_json(
        root / "role_neutral_handoff_binding.json",
        {**binding_body, "content_sha256": _sha(binding_body)},
    )
    paths = {execution, bundle, numerical, locator, binding}
    result = terminal_module._validate_portable_stage1_handoff_binding(
        request={"request_sha256": "1" * 64},
        stage1_paths=paths,
        bundle_path=bundle,
        bundle=bundle_value,
        numerical_manifest_path=numerical,
        numerical_identity={"manifest_content_sha256": "6" * 64},
    )
    assert result["closed_terminal_inventory_validated"] is True

    unrelated = _write_json(root / "legacy_handoff.json", {"legacy": True})
    with pytest.raises(ValueError, match="unrelated terminal artifact"):
        terminal_module._validate_portable_stage1_handoff_binding(
            request={"request_sha256": "1" * 64},
            stage1_paths=paths | {unrelated},
            bundle_path=bundle,
            bundle=bundle_value,
            numerical_manifest_path=numerical,
            numerical_identity={"manifest_content_sha256": "6" * 64},
        )


def _direct_handoff_validation_fixture(tmp_path: Path):
    bundle = _write_json(
        tmp_path / "direct_stage1" / "bundle_manifest.json",
        {"fixture": "reference-only"},
    )
    row_map = tmp_path / "direct_stage1" / "row_registry.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": list(range(10)),
            "id": [f"p{row_id}" for row_id in range(10)],
        }
    ).to_parquet(row_map, index=False)
    row_map = row_map.resolve()
    row_map_sha, _size = stable_file_sha256(row_map)
    assignments = {
        fold: {
            "fit_row_ids": [
                row_id for row_id in range(10) if row_id not in {2 * (fold - 1), 2 * (fold - 1) + 1}
            ],
            "heldout_row_ids": [
                2 * (fold - 1),
                2 * (fold - 1) + 1,
            ],
        }
        for fold in range(1, 6)
    }
    return {
        "real_stage1_handoff_detected": True,
        "bundle_manifest_path": str(bundle),
        "bundle_sha256": "1" * 64,
        "handoff_content_sha256": "2" * 64,
        "handoff_kind": ("authenticated_role_neutral_all_ten_reference_only_v1"),
        "handoff_scientific_content_sha256": "3" * 64,
        "source_role_neutral_execution_content_sha256": "4" * 64,
        "stage2_provider_identity_sha256": "5" * 64,
        "scope_plan_scientific_content_sha256": "6" * 64,
        "row_map_path": str(row_map),
        "row_map_sha256": row_map_sha,
        "direct_numerical_bank_content_sha256": "7" * 64,
        "outer_fold_assignments": assignments,
    }


def _direct_canary_fixture(
    tmp_path: Path,
    *,
    request,
    handoff,
):
    from scripts import (
        canary_production_stage1_hierarchy as canary_module,
    )

    guard = _prompt_guard_identity(tmp_path)
    implementation_sha, _implementation_size = stable_file_sha256(
        Path(canary_module.__file__).resolve(strict=True)
    )
    request_sha = "8" * 64
    prompt_audit = _prompt_audit(
        guard_sha=guard["identity_sha256"],
        request_sha=request_sha,
        generation_tokens=request["stage2_prompt_protocol"]["proposal_max_tokens"],
    )
    protocol = {
        "schema_version": "stage2_hierarchy_prompt_protocol_v3",
        **request["stage2_prompt_protocol"],
    }
    transport = {
        "job_id": "direct-canary",
        "job_kind": "interpret_evidence_chunk",
        "request_sha256": request_sha,
        "runner_identity_sha256": "9" * 64,
        "outcome": "success",
        "parsed_response_sha256": "a" * 64,
        "attempts": [
            {
                "attempt_number": 1,
                "endpoint": request["endpoint"],
                "model": request["model_name"],
                "request_sha256": request_sha,
                "runner_identity_sha256": "9" * 64,
                "response_model": request["model_name"],
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": prompt_audit["endpoint_prompt_tokens"],
                    "completion_tokens": 1,
                    "total_tokens": (prompt_audit["endpoint_prompt_tokens"] + 1),
                },
                "prompt_nontruncation_audit": prompt_audit,
                "outcome": "success",
                "retryable": False,
                "will_retry": False,
            }
        ],
    }
    selected = {
        "selection_order": [
            "rendered_message_bytes",
            "outer_fold",
            "job_id",
        ],
        "outer_fold": 1,
        "source_family": "bow_nuisance",
        "scope": "outer_fold_001_initial",
        "chunk_id_sha256": "b" * 64,
        "job_id": "direct-canary",
        "job_sha256": "c" * 64,
        "rendered_message_bytes": 1234,
        "evidence_owner_count": 2,
        "evidence_owner_ids_sha256": "d" * 64,
        "semantic_member_count": 3,
        "response_schema_sha256": "e" * 64,
        "identifier_ownership_sha256": "f" * 64,
        "response_contract_binding_sha256": "1" * 64,
        "local_json_schema_validator_identity_sha256": "2" * 64,
    }
    body = {
        "status": "accepted",
        "canary_kind": ("one_real_architecture_pure_initial_interpretation_job"),
        "authorization_role": "non_authorizing_operational_runtime_check",
        "stage1_bundle": {
            "manifest_path": handoff["bundle_manifest_path"],
            "handoff_kind": handoff["handoff_kind"],
            "bundle_sha256": handoff["bundle_sha256"],
            "handoff_content_sha256": handoff["handoff_scientific_content_sha256"],
            "source_execution_content_sha256": handoff[
                "source_role_neutral_execution_content_sha256"
            ],
            "provider_identity_sha256": handoff["stage2_provider_identity_sha256"],
            "reference_only_all_ten": True,
            "legacy_stage1_loader_invoked": False,
            "independent_stage1_refit_performed": False,
        },
        "endpoint": request["endpoint"],
        "model": request["model_name"],
        "runner_identity_sha256": "9" * 64,
        "settings": {
            "proposal_max_tokens": protocol["proposal_max_tokens"],
            "extraction_max_tokens": protocol["extraction_max_tokens"],
            "stage2_hierarchy_prompt_protocol": protocol,
            "stage2_hierarchy_prompt_protocol_sha256": _sha(protocol),
            "post_extraction_causal_review": request["post_extraction_causal_review"],
            "post_extraction_causal_review_sha256": _sha(request["post_extraction_causal_review"]),
            "prompt_nontruncation_guard_identity_sha256": guard["identity_sha256"],
            "transport_retries": 0,
            "selector_thinking_enabled": True,
            "selector_thinking_token_budget": protocol["selector_thinking_token_budget"],
            "max_rendered_discovery_prompt_bytes": protocol["max_rendered_discovery_prompt_bytes"],
            "final_upstream_max_orphan_features": protocol["final_upstream_max_orphan_features"],
            "review_neural_query_nuisance_folds": protocol["review_neural_query_nuisance_folds"],
            "final_upstream_meta_inner_folds": protocol["final_upstream_meta_inner_folds"],
            "final_upstream_head_regularization": protocol["final_upstream_head_regularization"],
            "extraction_thinking_enabled": False,
            "maximum_schema_repairs": 1,
        },
        "selected_job": selected,
        "validation": {
            "normalized_response_sha256": "3" * 64,
            "raw_wire_response_sha256": "4" * 64,
            "response_attempt_trace_sha256": "5" * 64,
            "response_attempt_outcomes": ["validated_response"],
            "local_json_schema_validator_identity_sha256": "6" * 64,
            "response_repair_policy_sha256": "7" * 64,
            "job_cache_identity_sha256": "8" * 64,
            "validated_only_cache_enabled": True,
        },
        "remote_response_count": 1,
        "transport_metadata": [transport],
        "raw_prompt_emitted": False,
        "raw_response_emitted": False,
        "normalized_findings_emitted": False,
        "prediction_path_constructed": False,
        "oracle_path_constructed": False,
        "full_fusion_runner_executed": False,
        "reference_only_role_neutral_stage1": True,
        "legacy_stage1_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "canary_implementation_file_sha256": implementation_sha,
    }
    return (
        _wrapped(
            tmp_path / "direct_canary" / "production_role_neutral_stage2_runtime_canary.json",
            schema="production_role_neutral_stage2_runtime_canary_report_v1",
            body=body,
        ),
        guard,
    )


def test_direct_canary_dispatch_rejects_legacy_substitution(tmp_path):
    request = {
        **_portable_forest_request(tmp_path),
        "outer_folds": 5,
    }
    handoff = _direct_handoff_validation_fixture(tmp_path)
    canary, _guard = _direct_canary_fixture(
        tmp_path,
        request=request,
        handoff=handoff,
    )
    records = [_record("stage2_canary", [canary])]
    result = validate_real_stage2_canary(
        request=request,
        phase_records=records,
        handoff_validation=handoff,
    )
    assert result["reference_only_role_neutral_stage1"] is True
    assert result["finish_reason_stop_proven"] is True

    wrapper = json.loads(canary.read_text(encoding="utf-8"))
    wrapper["body"]["legacy_stage1_loader_invoked"] = True
    _rewrite_wrapper_content_hash(canary, wrapper)
    records = [_record("stage2_canary", [canary])]
    with pytest.raises(ValueError, match="scientific/transport"):
        validate_real_stage2_canary(
            request=request,
            phase_records=records,
            handoff_validation=handoff,
        )


def _complete_paged_ledger_fixture(
    inference_root: Path,
    *,
    request,
    prepared_path: Path,
    guard_sha: str,
):
    from oci.extraction.complete_paged import (
        COMPLETE_PAGED_RESPONSE_SCHEMA,
        COMPLETE_PAGED_TRANSPORT_SCHEMA,
        CompleteFeatureContract,
        CompletePageResponse,
        CompletePagingGeometry,
        build_complete_page_prompt,
        build_complete_paged_coverage_ledger,
        plan_complete_paged_requests,
        reconcile_complete_page_responses,
    )
    from oci.inference.production_stage1_hierarchy_one_shot import (
        PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA,
    )

    prepared = pd.read_parquet(prepared_path)
    ordered_ids = [int(value) for value in prepared["_oci_row_id"].tolist()]
    texts = prepared[request["text_column"]].tolist()
    geometry = CompletePagingGeometry(
        core_chars=request["complete_page_core_chars"],
        context_chars=request["complete_page_context_chars"],
        max_page_chars=request["complete_page_max_chars"],
    )
    feature = CompleteFeatureContract(
        name="documented_biomarker",
        value_type="categorical",
        description="Whether the prepared note documents the biomarker",
        temporal_rule="use_only_complete_prepared_decision_time_text",
        aggregation_rule="reconcile_all_pages_without_loss",
        categories=("documented", "not_documented"),
    )
    feature_contract = {
        "name": feature.name,
        "value_type": feature.value_type,
        "description": feature.description,
        "temporal_rule": feature.temporal_rule,
        "aggregation_rule": feature.aggregation_rule,
        "categories": list(feature.categories),
    }
    notes = {str(index): text for index, text in enumerate(texts)}
    plan = plan_complete_paged_requests(
        notes,
        (feature,),
        geometry=geometry,
    )
    normalized_responses = {}
    response_objects = {}
    transport_audits = {}
    prompts = {}
    page_rows = []
    for request_index, page_request in enumerate(plan.requests):
        patient_index = int(page_request.patient_id)
        prompt = build_complete_page_prompt(
            texts[patient_index],
            page=page_request.page,
            feature=feature,
            geometry=geometry,
        )
        response = CompletePageResponse.validate(
            {
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "negative",
                "normalized_value": "not_documented",
                "reason": None,
                "citations": [],
            },
            text=texts[patient_index],
            page=page_request.page,
        )
        normalized = response.as_dict()
        initial_request = {
            "model": request["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": request["stage2_prompt_protocol"]["extraction_max_tokens"],
        }
        attempt = {
            "kind": "initial",
            "request_sha256": _sha(initial_request),
            "response_sha256": _sha(normalized),
            "model": request["model_name"],
            "finish_reason": "stop",
        }
        transport_body = {
            "schema_version": COMPLETE_PAGED_TRANSPORT_SCHEMA,
            "transport_retry_count": 0,
            "schema_repair_count": 0,
            "configured_model": request["model_name"],
            "attempts": [attempt],
        }
        transport = {
            **transport_body,
            "content_sha256": _sha(transport_body),
        }
        normalized_responses[page_request.request_id] = normalized
        response_objects[page_request.request_id] = response
        transport_audits[page_request.request_id] = transport
        prompts[page_request.request_id] = prompt
        page_rows.append(
            {
                "request_index": request_index,
                "request_id": page_request.request_id,
                "patient_local_id": page_request.patient_id,
                "oci_row_id": ordered_ids[patient_index],
                "note_sha256": page_request.note_sha256,
                "feature_name": page_request.feature_name,
                "feature_contract_sha256": (page_request.feature_contract_sha256),
                "page_index": page_request.page.page_index,
                "core_start": page_request.page.core_start,
                "core_end": page_request.page.core_end,
                "context_start": page_request.page.context_start,
                "context_end": page_request.page.context_end,
                "page_text_sha256": page_request.page.text_sha256,
                "core_sha256": page_request.page.core_sha256,
                "prompt_sha256": page_request.prompt_sha256,
                "prompt": prompt,
                "normalized_response_json": json.dumps(
                    normalized,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "normalized_response_sha256": _sha(normalized),
                "transport_audit_json": json.dumps(
                    transport,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "transport_audit_sha256": _sha(transport),
            }
        )

    reconciliation_rows = []
    for patient_index, text in enumerate(texts):
        patient_id = str(patient_index)
        patient_requests = [
            page_request for page_request in plan.requests if page_request.patient_id == patient_id
        ]

        def no_reduction_needed(_children):
            raise AssertionError("the one-page fixture must not need a reconciliation call")

        final, reconciliation = reconcile_complete_page_responses(
            [
                (
                    page_request.request_id,
                    response_objects[page_request.request_id],
                )
                for page_request in patient_requests
            ],
            reducer=no_reduction_needed,
            fan_in=request["complete_reconciliation_fan_in"],
        )
        final_response = final.as_dict()
        reconciliation_rows.append(
            {
                "patient_local_id": patient_id,
                "oci_row_id": ordered_ids[patient_index],
                "final_response_json": json.dumps(
                    final_response,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "final_response_sha256": _sha(final_response),
                "reconciliation_ledger_json": json.dumps(
                    reconciliation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "reconciliation_ledger_sha256": _sha(reconciliation),
                "transport_audits_json": "[]",
                "transport_audits_sha256": _sha([]),
            }
        )

    coverage = build_complete_paged_coverage_ledger(
        plan,
        normalized_responses,
    )
    ledger_root = inference_root / "complete_paged_extraction_ledgers" / "ledger_fixture"
    ledger_root.mkdir(parents=True)
    page_path = ledger_root / "page_requests.parquet"
    reconciliation_path = ledger_root / "reconciliation.parquet"
    pd.DataFrame(page_rows).to_parquet(page_path, index=False)
    pd.DataFrame(reconciliation_rows).to_parquet(
        reconciliation_path,
        index=False,
    )
    page_path = page_path.resolve()
    reconciliation_path = reconciliation_path.resolve()
    page_sha, page_size = stable_file_sha256(page_path)
    reconciliation_sha, reconciliation_size = stable_file_sha256(reconciliation_path)
    geometry_value = geometry.as_dict()
    manifest_body = {
        "schema_version": (PRODUCTION_COMPLETE_PAGED_EXTRACTION_LEDGER_SCHEMA),
        "feature_contract": feature_contract,
        "feature_contract_sha256": feature.contract_sha256,
        "configured_model": request["model_name"],
        "geometry": geometry_value,
        "geometry_sha256": _sha(geometry_value),
        "ordered_oci_row_ids": ordered_ids,
        "ordered_oci_row_ids_sha256": _sha(ordered_ids),
        "ordered_note_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
        "request_plan_content_sha256": plan.as_dict()["content_sha256"],
        "coverage_content_sha256": coverage["content_sha256"],
        "planned_page_request_count": len(plan.requests),
        "completed_page_request_count": len(plan.requests),
        "patient_count": len(texts),
        "page_table": {
            "relative_path": page_path.name,
            "row_count": len(page_rows),
            "size": page_size,
            "sha256": page_sha,
        },
        "reconciliation_table": {
            "relative_path": reconciliation_path.name,
            "row_count": len(reconciliation_rows),
            "size": reconciliation_size,
            "sha256": reconciliation_sha,
        },
        "one_feature_contract_per_page_request": True,
        "configured_reconciliation_fan_in": request["complete_reconciliation_fan_in"],
        "all_pages_reconciled_with_configured_fan_in": True,
        "transport_retries": 0,
        "maximum_schema_repairs_per_request": 1,
        "exact_prompts_persisted": True,
        "canonical_row_ids_persisted": True,
        "raw_note_copies_persisted": False,
    }
    manifest_path = _write_json(
        ledger_root / "manifest.json",
        {
            **manifest_body,
            "content_sha256": _sha(manifest_body),
        },
    )
    manifest_sha, manifest_size = stable_file_sha256(manifest_path)
    registration = {
        "invocation_index": 0,
        "manifest": {
            "path": str(manifest_path),
            "size": manifest_size,
            "sha256": manifest_sha,
            "content_sha256": _sha(manifest_body),
        },
        "payloads": [
            {
                "kind": "page_table",
                "path": str(page_path),
                "size": page_size,
                "sha256": page_sha,
            },
            {
                "kind": "reconciliation_table",
                "path": str(reconciliation_path),
                "size": reconciliation_size,
                "sha256": reconciliation_sha,
            },
        ],
    }
    inventory = [
        {
            "kind": "complete_paged_page_table",
            "invocation_index": 0,
            "path": str(page_path),
            "size": page_size,
            "sha256": page_sha,
        },
        {
            "kind": "complete_paged_reconciliation_table",
            "invocation_index": 0,
            "path": str(reconciliation_path),
            "size": reconciliation_size,
            "sha256": reconciliation_sha,
        },
        {
            "kind": "complete_paged_ledger_manifest",
            "invocation_index": 0,
            **registration["manifest"],
        },
    ]
    prompt_records = [
        _prompt_audit(
            guard_sha=guard_sha,
            request_sha=transport_audits[page_request.request_id]["attempts"][0]["request_sha256"],
            generation_tokens=request["stage2_prompt_protocol"]["extraction_max_tokens"],
            client_path="explicit_feature_extraction",
        )
        for page_request in plan.requests
    ]
    assert len(prompt_records) == len(texts) == 10
    assert len(prompts) == len(normalized_responses) == 10
    return registration, inventory, prompt_records


def _direct_terminal_fixture(tmp_path: Path, *, oracle: bool):
    from oci.inference import (
        production_stage1_hierarchy_one_shot as one_shot_module,
    )
    from oci.inference.all_evidence_fusion_runner import (
        _numerical_array_sha256,
    )
    from oci.inference.final_context_fit_causal_forest_adapter import (
        _repository_causal_forest_runtime_attestation,
    )
    from oci.inference.fold_honest_signal_fusion import (
        row_set_fingerprint,
    )

    handoff = _direct_handoff_validation_fixture(tmp_path)
    repository_runtime = dict(_repository_causal_forest_runtime_attestation())
    base_request, _runtime, fold_template = _portable_forest_fold(
        tmp_path,
        runtime_override=repository_runtime,
    )
    request = {
        **base_request,
        "outer_folds": 5,
        "evaluate_oracle_posthoc": oracle,
        "text_column": "prepared_text",
        "complete_page_core_chars": 160,
        "complete_page_context_chars": 12,
        "complete_page_max_chars": 184,
        "complete_reconciliation_fan_in": 4,
    }
    canary_path, guard = _direct_canary_fixture(
        tmp_path,
        request=request,
        handoff=handoff,
    )
    inference_root = (tmp_path / "direct_inference").resolve()
    inference_root.mkdir(parents=True)
    prepared_path = inference_root / "authenticated_prepared_cohort.parquet"
    pd.DataFrame(
        {
            "_oci_row_id": list(range(10)),
            "prepared_text": [
                (
                    f"Prepared decision-time note for patient p{row_id}. "
                    "No qualifying biomarker is documented."
                )
                for row_id in range(10)
            ],
        }
    ).to_parquet(prepared_path, index=False)
    prepared_path = prepared_path.resolve()
    prepared_sha, prepared_size = stable_file_sha256(prepared_path)
    prepared_registration = {
        "path": str(prepared_path),
        "size": prepared_size,
        "sha256": prepared_sha,
        "row_count": 10,
        "text_column": request["text_column"],
    }
    source = {
        "mode": handoff["handoff_kind"],
        "provider_identity_sha256": handoff["stage2_provider_identity_sha256"],
        "plan_scientific_content_sha256": handoff["scope_plan_scientific_content_sha256"],
        "source_execution_content_sha256": handoff["source_role_neutral_execution_content_sha256"],
        "runtime_binding_content_sha256": "8" * 64,
        "prepared_projection_binding_content_sha256": "9" * 64,
        "prepared_cohort_artifact_sha256": prepared_sha,
        "row_map_sha256": handoff["row_map_sha256"],
        "direct_numerical_bank_manifest_content_sha256": handoff[
            "direct_numerical_bank_content_sha256"
        ],
        "legacy_stage1_loader_invoked": False,
        "tfidf_handoff_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "text_truncation_applied": False,
    }
    combined = pd.DataFrame(
        {
            "_oci_row_id": list(range(10)),
            "outer_fold": [fold for fold in range(1, 6) for _row in range(2)],
            "pred_ite_prob": np.linspace(-0.4, 0.4, 10),
        }
    )
    combined_path = inference_root / "frozen_predictions.parquet"
    combined.to_parquet(combined_path, index=False)
    combined_path = combined_path.resolve()
    combined_sha, combined_size = stable_file_sha256(combined_path)

    fold_manifest_paths = []
    fold_prediction_paths = []
    fold_attestations = []
    phase_inventory = []
    for fold in range(1, 6):
        assignment = handoff["outer_fold_assignments"][fold]
        frame = combined[combined["outer_fold"] == fold].reset_index(drop=True)
        fold_dir = inference_root / f"fold_{fold:03d}"
        fold_dir.mkdir()
        prediction_path = fold_dir / "frozen_predictions.parquet"
        frame.to_parquet(prediction_path, index=False)
        prediction_path = prediction_path.resolve()
        prediction_sha, prediction_size = stable_file_sha256(prediction_path)

        estimator = json.loads(json.dumps(fold_template["final_ite_estimator"]))
        receipt = estimator["forest_receipt"]
        receipt["outer_fold"] = fold
        receipt["reference_manifest_content_sha256"] = handoff[
            "direct_numerical_bank_content_sha256"
        ]
        receipt["fit_row_count"] = len(assignment["fit_row_ids"])
        receipt["prediction_row_count"] = len(assignment["heldout_row_ids"])
        receipt["tau_sha256"] = _numerical_array_sha256(
            frame["pred_ite_prob"].to_numpy(dtype=float)
        )
        receipt_body = {key: value for key, value in receipt.items() if key != "content_sha256"}
        receipt["content_sha256"] = _sha(receipt_body)
        body = {
            "outer_fold": fold,
            "train_row_count": len(assignment["fit_row_ids"]),
            "heldout_row_count": len(assignment["heldout_row_ids"]),
            "train_row_fingerprint": row_set_fingerprint(assignment["fit_row_ids"]),
            "heldout_row_fingerprint": row_set_fingerprint(assignment["heldout_row_ids"]),
            "outer_heldout_outcomes_used": False,
            "oracle_columns_written": False,
            "prediction_columns": list(frame.columns),
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "legacy_handoff_sha256": None,
            "tfidf_handoff_sha256": None,
            "stage1_reference_source": source,
            "final_ite_estimator": estimator,
        }
        manifest_path = _wrapped(
            fold_dir / "immutable_fold_manifest.json",
            schema="all_evidence_fusion_frozen_fold_v20",
            body=body,
        )
        manifest_sha, manifest_size = stable_file_sha256(manifest_path)
        manifest_content_sha = json.loads(manifest_path.read_text(encoding="utf-8"))[
            "content_sha256"
        ]
        manifest_registration = {
            "path": str(manifest_path),
            "size": manifest_size,
            "sha256": manifest_sha,
            "content_sha256": manifest_content_sha,
        }
        prediction_registration = {
            "path": str(prediction_path),
            "size": prediction_size,
            "sha256": prediction_sha,
        }
        fold_manifest_paths.append(manifest_path)
        fold_prediction_paths.append(prediction_path)
        fold_attestations.append(
            {
                "outer_fold": fold,
                "fit_row_count": len(assignment["fit_row_ids"]),
                "heldout_row_count": len(assignment["heldout_row_ids"]),
                "manifest": manifest_registration,
                "prediction": prediction_registration,
                "strict_forest_receipt_content_sha256": receipt["content_sha256"],
            }
        )
        phase_inventory.extend(
            (
                {
                    "kind": "fold_manifest",
                    "outer_fold": fold,
                    **manifest_registration,
                },
                {
                    "kind": "fold_prediction",
                    "outer_fold": fold,
                    **prediction_registration,
                },
            )
        )

    batch_path = _wrapped(
        inference_root / "preparation" / "authenticated_hierarchical_batch_result.json",
        schema="hierarchical_all_evidence_runner_batch_result_v1",
        body={
            "ordered_fold_results": [{"outer_fold": fold} for fold in range(1, 6)],
            "all_fold_discovery_completed_before_per_fold_modeling": True,
        },
    )
    batch_sha, batch_size = stable_file_sha256(batch_path)
    batch_content_sha = json.loads(batch_path.read_text(encoding="utf-8"))["content_sha256"]
    batch_registration = {
        "path": str(batch_path),
        "size": batch_size,
        "sha256": batch_sha,
        "content_sha256": batch_content_sha,
        "all_fold_discovery_completed_before_per_fold_modeling": True,
    }
    input_path = _wrapped(
        inference_root / "immutable_input_manifest.json",
        schema="all_evidence_fusion_outer_runner_v20",
        body={
            "stage1_reference_source": source,
            "legacy_handoff_path": None,
            "legacy_handoff_sha256": None,
            "tfidf_handoff_path": None,
            "tfidf_handoff_sha256": None,
        },
    )
    input_sha, input_size = stable_file_sha256(input_path)
    input_content_sha = json.loads(input_path.read_text(encoding="utf-8"))["content_sha256"]
    input_registration = {
        "path": str(input_path),
        "size": input_size,
        "sha256": input_sha,
        "content_sha256": input_content_sha,
    }
    run_path = _wrapped(
        inference_root / "immutable_run_manifest.json",
        schema="all_evidence_fusion_predictions_v5",
        body={
            "fold_manifest_paths": [str(path) for path in fold_manifest_paths],
            "fold_count": 5,
            "prediction_path": str(combined_path),
            "prediction_sha256": combined_sha,
            "prediction_row_count": len(combined),
            "prediction_columns": list(combined.columns),
            "outer_test_rows_predicted_once": True,
            "final_ite_estimator": {
                "mode": ("strict_outer_honest_final_context_fit_causal_forest_v2"),
                "strict_causal_forest_active_for_every_fold": True,
                "strict_causal_forest_required": True,
                "fixed_prior_working_backend_active": True,
                "reference_only_role_neutral_runtime": True,
            },
            "oracle_columns_written": False,
        },
    )
    run_sha, run_size = stable_file_sha256(run_path)
    run_content_sha = json.loads(run_path.read_text(encoding="utf-8"))["content_sha256"]
    protocol = {
        "schema_version": "stage2_hierarchy_prompt_protocol_v3",
        **request["stage2_prompt_protocol"],
    }
    (
        ledger_registration,
        ledger_inventory,
        extraction_prompt_records,
    ) = _complete_paged_ledger_fixture(
        inference_root,
        request=request,
        prepared_path=prepared_path,
        guard_sha=guard["identity_sha256"],
    )
    prompt_records = [
        _prompt_audit(
            guard_sha=guard["identity_sha256"],
            request_sha="b" * 64,
            generation_tokens=protocol["proposal_max_tokens"],
            client_path="hierarchical_discovery",
        ),
        _prompt_audit(
            guard_sha=guard["identity_sha256"],
            request_sha="c" * 64,
            generation_tokens=protocol["proposal_max_tokens"],
            client_path="proposal_and_post_extraction_review",
        ),
        *extraction_prompt_records,
    ]
    implementation_sha, _implementation_size = stable_file_sha256(
        Path(one_shot_module.__file__).resolve(strict=True)
    )
    phase_inventory.extend(ledger_inventory)
    phase_inventory.append(
        {
            "kind": "prepared_cohort",
            **{key: prepared_registration[key] for key in ("path", "size", "sha256")},
        }
    )
    phase_inventory.extend(
        (
            {
                "kind": "hierarchical_batch_result",
                "path": str(batch_path),
                "size": batch_size,
                "sha256": batch_sha,
                "content_sha256": batch_content_sha,
            },
            {
                "kind": "runner_input_manifest",
                **input_registration,
            },
            {
                "kind": "combined_prediction",
                "path": str(combined_path),
                "size": combined_size,
                "sha256": combined_sha,
            },
            {
                "kind": "run_manifest",
                "path": str(run_path),
                "size": run_size,
                "sha256": run_sha,
                "content_sha256": run_content_sha,
            },
        )
    )
    attestation_body = {
        "schema_version": ("production_role_neutral_stage2_one_shot_attestation_v2"),
        "status": "completed",
        "handoff_kind": handoff["handoff_kind"],
        "stage1_reference_handoff": {
            "manifest_path": handoff["bundle_manifest_path"],
            "scientific_content_sha256": handoff["handoff_scientific_content_sha256"],
            "bundle_sha256": handoff["bundle_sha256"],
            "source_execution_content_sha256": handoff[
                "source_role_neutral_execution_content_sha256"
            ],
            "provider_identity_sha256": handoff["stage2_provider_identity_sha256"],
            "runtime_binding_content_sha256": source["runtime_binding_content_sha256"],
            "prepared_projection_binding_content_sha256": source[
                "prepared_projection_binding_content_sha256"
            ],
            "prepared_cohort_artifact_sha256": source["prepared_cohort_artifact_sha256"],
            "row_map_sha256": handoff["row_map_sha256"],
            "direct_numerical_bank_manifest_content_sha256": handoff[
                "direct_numerical_bank_content_sha256"
            ],
            "offline_handoff_validation_complete": True,
        },
        "remote_runtime_identity": {
            "endpoint_urls": [request["endpoint"]],
            "model": {"name": request["model_name"]},
            "hierarchical_runner_identity_sha256": "e" * 64,
            "prompt_nontruncation_guard": guard,
            "prompt_nontruncation_execution_audit": (
                _prompt_execution_audit(
                    guard_sha=guard["identity_sha256"],
                    records=prompt_records,
                )
            ),
            "required_finish_reason": "stop",
            "endpoint_pool_or_fallback_allowed": False,
            "model_substitution_allowed": False,
        },
        "stage2_hierarchy_prompt_protocol": protocol,
        "post_extraction_causal_review": request["post_extraction_causal_review"],
        "hierarchical_batch_result": batch_registration,
        "folds": fold_attestations,
        "fold_count": 5,
        "runner_input_manifest": input_registration,
        "prepared_cohort": prepared_registration,
        "complete_paged_extraction_ledgers": [ledger_registration],
        "immutable_run_manifest": {
            "path": str(run_path),
            "sha256": run_sha,
            "content_sha256": run_content_sha,
        },
        "frozen_predictions": {
            "path": str(combined_path),
            "size": combined_size,
            "sha256": combined_sha,
            "columns": list(combined.columns),
            "row_count": len(combined),
            "probability_difference_bounds": [-1.0, 1.0],
            "probability_difference_validation_tolerance": float(64 * np.finfo(np.float64).eps),
            "probability_difference_bounds_validated": True,
            "values_clipped": False,
        },
        "phase_artifact_inventory": phase_inventory,
        "one_shot_implementation_sha256": implementation_sha,
        "legacy_stage1_loader_invoked": False,
        "tfidf_handoff_loader_invoked": False,
        "independent_stage1_refit_performed": False,
        "structured_or_nonforest_fallback_used": False,
        "outer_heldout_labels_used_during_discovery_or_review": False,
        "oracle_source_opened": False,
        "global_release_certified": False,
    }
    attestation_path = _write_json(
        inference_root / "attestation" / "production_role_neutral_stage2_one_shot_result.json",
        {
            **attestation_body,
            "content_sha256": _sha(attestation_body),
        },
    )
    handoff_report = _write_json(
        tmp_path / "direct_handoff_validation.json",
        {"status": "accepted"},
    )
    inference_paths = [
        *fold_manifest_paths,
        *fold_prediction_paths,
        prepared_path,
        Path(ledger_registration["manifest"]["path"]),
        *[Path(payload["path"]) for payload in ledger_registration["payloads"]],
        batch_path,
        input_path,
        combined_path,
        run_path,
        attestation_path,
    ]
    records = [
        _record(
            "stage1_modeling",
            [
                Path(handoff["bundle_manifest_path"]),
                Path(handoff["row_map_path"]),
            ],
        ),
        _record("handoff_validation", [handoff_report]),
        _record("stage2_canary", [canary_path]),
        _record("stage2_inference", inference_paths),
    ]
    if oracle:
        oracle_path = tmp_path / "direct_oracle.parquet"
        pd.DataFrame(
            {
                "id": [f"p{row_id}" for row_id in range(10)],
                "true_ite_prob": np.linspace(-0.3, 0.5, 10),
            }
        ).to_parquet(oracle_path, index=False)
        evaluation = evaluate_frozen_predictions_posthoc(
            predictions_path=combined_path,
            prediction_manifest_path=run_path,
            unit_id_map_path=Path(handoff["row_map_path"]),
            oracle_dataset_path=oracle_path,
            output_dir=tmp_path / "direct_evaluation",
            unit_id_column="id",
            oracle_unit_id_column="id",
            oracle_ite_column="true_ite_prob",
        )
        records.append(
            _record(
                "oracle_evaluation",
                [
                    Path(evaluation["joined_path"]),
                    tmp_path / "direct_evaluation" / "evaluation_metrics.json",
                ],
            )
        )
    else:
        records.append(_record("oracle_evaluation", []))
    return request, records, handoff, attestation_path


def test_direct_terminal_reopens_all_five_strict_folds_and_closed_inventory(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, _attestation = _direct_terminal_fixture(tmp_path, oracle=False)
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    result = validate_real_stage2_terminal_artifacts(
        request=request,
        phase_records=records,
    )
    assert result["portable_reference_only_stage2_validated"] is True
    assert result["fold_manifest_count"] == 5
    assert result["fold_prediction_count"] == 5
    assert result["strict_forest_identity_validated_per_fold"] is True
    assert result["row_order_validated"] is True
    assert result["workflow_phase_order_validated"] is True
    assert result["global_release_certified"] is False
    one_shot = result["stage2_one_shot_validation"]
    assert one_shot["prompt_nontruncation_execution_record_count"] == 12
    assert one_shot["complete_paged_planned_page_request_count"] == 10
    assert one_shot["complete_paged_remote_request_count"] == 10
    assert one_shot["complete_paged_exact_prompt_and_citation_validation"] is True
    assert len(one_shot["complete_paged_ledger_manifest_paths"]) == 1
    assert len(one_shot["complete_paged_ledger_payload_paths"]) == 2


def _reseal_direct_ledger_bindings(
    *,
    records,
    attestation_path: Path,
) -> None:
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    ledger = attestation["complete_paged_extraction_ledgers"][0]
    manifest_path = Path(ledger["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_by_kind = {row["kind"]: row for row in ledger["payloads"]}
    for kind, manifest_field in (
        ("page_table", "page_table"),
        ("reconciliation_table", "reconciliation_table"),
    ):
        registration = payload_by_kind[kind]
        digest, size = stable_file_sha256(Path(registration["path"]))
        registration["sha256"] = digest
        registration["size"] = size
        manifest[manifest_field]["sha256"] = digest
        manifest[manifest_field]["size"] = size
    _rewrite_flat_content_hash(manifest_path, manifest)
    manifest_digest, manifest_size = stable_file_sha256(manifest_path)
    ledger["manifest"].update(
        {
            "sha256": manifest_digest,
            "size": manifest_size,
            "content_sha256": manifest["content_sha256"],
        }
    )
    inventory_by_kind = {
        row["kind"]: row
        for row in attestation["phase_artifact_inventory"]
        if str(row["kind"]).startswith("complete_paged_")
    }
    for kind, inventory_kind in (
        ("page_table", "complete_paged_page_table"),
        (
            "reconciliation_table",
            "complete_paged_reconciliation_table",
        ),
    ):
        registration = payload_by_kind[kind]
        inventory_by_kind[inventory_kind].update(
            {
                "size": registration["size"],
                "sha256": registration["sha256"],
            }
        )
    inventory_by_kind["complete_paged_ledger_manifest"].update(
        {
            "size": manifest_size,
            "sha256": manifest_digest,
            "content_sha256": manifest["content_sha256"],
        }
    )
    _rewrite_flat_content_hash(attestation_path, attestation)
    _refresh_record(_record_for_phase(records, "stage2_inference"))


def test_direct_terminal_rejects_changed_complete_paged_page_bytes(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(tmp_path, oracle=False)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    page_path = Path(attestation["complete_paged_extraction_ledgers"][0]["payloads"][0]["path"])
    page_path.write_bytes(page_path.read_bytes() + b"tampered")
    _refresh_record(_record_for_phase(records, "stage2_inference"))
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(ValueError, match="page_table bytes changed"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_terminal_reconstructs_and_rejects_changed_page_prompt(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(tmp_path, oracle=False)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    page_path = Path(attestation["complete_paged_extraction_ledgers"][0]["payloads"][0]["path"])
    frame = pd.read_parquet(page_path)
    frame.loc[0, "prompt"] = frame.loc[0, "prompt"] + "\nchanged"
    frame.to_parquet(page_path, index=False)
    _reseal_direct_ledger_bindings(
        records=records,
        attestation_path=attestation_path,
    )
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(ValueError, match="request/prompt changed"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_terminal_rejects_resealed_complete_page_request_identity_substitution(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(
        tmp_path,
        oracle=False,
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    page_path = Path(
        attestation["complete_paged_extraction_ledgers"][0]["payloads"][0]["path"]
    )
    frame = pd.read_parquet(page_path)
    transport = json.loads(frame.loc[0, "transport_audit_json"])
    transport["attempts"][0]["request_sha256"] = "0" * 64
    transport_body = {
        key: value for key, value in transport.items() if key != "content_sha256"
    }
    transport["content_sha256"] = _sha(transport_body)
    frame.loc[0, "transport_audit_json"] = json.dumps(
        transport,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    frame.loc[0, "transport_audit_sha256"] = _sha(transport)
    frame.to_parquet(page_path, index=False)
    _reseal_direct_ledger_bindings(
        records=records,
        attestation_path=attestation_path,
    )
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(ValueError, match="count and request identity"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_terminal_recomputes_and_rejects_changed_coverage_proof(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(tmp_path, oracle=False)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    manifest_path = Path(attestation["complete_paged_extraction_ledgers"][0]["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage_content_sha256"] = "f" * 64
    _rewrite_flat_content_hash(manifest_path, manifest)
    _reseal_direct_ledger_bindings(
        records=records,
        attestation_path=attestation_path,
    )
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(ValueError, match="coverage proof changed"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_terminal_rejects_missing_complete_paged_ledger(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(tmp_path, oracle=False)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    ledger = attestation["complete_paged_extraction_ledgers"][0]
    ledger_paths = {
        Path(ledger["manifest"]["path"]),
        *[Path(row["path"]) for row in ledger["payloads"]],
    }
    attestation["complete_paged_extraction_ledgers"] = []
    attestation["phase_artifact_inventory"] = [
        row
        for row in attestation["phase_artifact_inventory"]
        if not str(row["kind"]).startswith("complete_paged_")
    ]
    _rewrite_flat_content_hash(attestation_path, attestation)
    inference_record = _record_for_phase(records, "stage2_inference")
    inference_record["artifacts"] = [
        row for row in inference_record["artifacts"] if Path(row["path"]) not in ledger_paths
    ]
    _refresh_record(inference_record)
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(
        ValueError,
        match="omitted complete-paged extraction ledgers",
    ):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_terminal_rejects_attestation_inventory_reordering(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, attestation_path = _direct_terminal_fixture(tmp_path, oracle=False)
    value = json.loads(attestation_path.read_text(encoding="utf-8"))
    value["phase_artifact_inventory"] = list(reversed(value["phase_artifact_inventory"]))
    _rewrite_flat_content_hash(attestation_path, value)
    _refresh_record(_record_for_phase(records, "stage2_inference"))
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    with pytest.raises(ValueError, match="artifact inventory"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )


def test_direct_oracle_phase_follows_graph_canary_five_folds_and_attestation(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, _attestation = _direct_terminal_fixture(tmp_path, oracle=True)
    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    result = validate_real_stage2_terminal_artifacts(
        request=request,
        phase_records=records,
    )
    oracle = result["oracle_validation"]
    assert oracle["oracle_open_order_proven"] is True
    assert oracle["workflow_phase_order_proven"] is True
    assert oracle["stage1_graph_handoff_and_canary_preceded_oracle"] is True
    assert oracle["all_configured_strict_folds_and_attestation_preceded_oracle"] is True
    assert oracle["configured_strict_fold_count_preceded_oracle"] == 5

    reordered = [
        *records[:3],
        records[-1],
        records[3],
    ]
    with pytest.raises(ValueError, match="before frozen Stage 2 inference"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=reordered,
        )


def test_direct_terminal_defers_oracle_artifact_reads_until_strict_folds(
    tmp_path,
    monkeypatch,
):
    request, records, handoff, _attestation = _direct_terminal_fixture(tmp_path, oracle=True)
    evaluation_record = _record_for_phase(records, "oracle_evaluation")
    evaluation_reopened = False
    original_artifact_paths = terminal_module._artifact_paths

    def tracked_artifact_paths(record):
        nonlocal evaluation_reopened
        if record is evaluation_record:
            evaluation_reopened = True
        return original_artifact_paths(record)

    def stop_at_strict_forest(**_kwargs):
        assert evaluation_reopened is False
        raise RuntimeError("strict-fold-order-sentinel")

    monkeypatch.setattr(
        terminal_module,
        "validate_real_stage1_handoff",
        lambda **_kwargs: handoff,
    )
    monkeypatch.setattr(
        terminal_module,
        "_artifact_paths",
        tracked_artifact_paths,
    )
    monkeypatch.setattr(
        terminal_module,
        "_validate_strict_forest",
        stop_at_strict_forest,
    )
    with pytest.raises(RuntimeError, match="strict-fold-order-sentinel"):
        validate_real_stage2_terminal_artifacts(
            request=request,
            phase_records=records,
        )
    assert evaluation_reopened is False
