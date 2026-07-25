from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import oci.inference.compact_preflight_compression_benchmark as compression_module
import oci.inference.role_neutral_benchmark_deployment_selection as selection_module
from oci.inference.compact_preflight_compression_benchmark import (
    COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA,
    COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA,
)
from oci.inference.portable_workflow_spec import (
    DeploymentProfile,
    ScientificWorkflowSpec,
    identity_sha256,
)
from oci.inference.production_text_preparation import stable_file_sha256
from oci.inference.role_neutral_benchmark_deployment_selection import (
    select_benchmarked_deployment_profile,
    validate_benchmarked_stage1_execution_profile,
)
from oci.inference.role_neutral_benchmark_workload_provider import (
    AuthenticatedPausedStage1Preflight,
    RoleNeutralBenchmarkScopeSelector,
    RoleNeutralBenchmarkWorkloadDeployment,
)
from oci.inference.role_neutral_performance_benchmark import (
    ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
    RoleNeutralBenchmarkConfig,
    build_role_neutral_benchmark_matrix_coverage,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_BASE_DEPLOYMENT = (
    _REPOSITORY
    / "example_configs"
    / "portable_all_evidence_deployment_nsclc.acceptance.json"
)
_SCIENTIFIC = (
    _REPOSITORY
    / "example_configs"
    / "portable_all_evidence_scientific_nsclc.json"
)
_BENCHMARK_CONFIG = (
    _REPOSITORY
    / "example_configs"
    / "portable_role_neutral_performance_benchmark_nsclc.deployment.json"
)


def _compression_result(
    config: RoleNeutralBenchmarkConfig,
) -> dict[str, Any]:
    compression_config = config.preflight_compression_benchmark
    schedule = compression_module._schedule(compression_config)
    observations: list[dict[str, Any]] = []
    scientific = "9" * 64
    source_storage = {
        "parquet_compression": "zstd",
        "registered_payload_bytes": 70,
        "manifest_bytes": 10,
        "tree_file_bytes": 80,
        "parquet_file_bytes": 50,
        "json_file_bytes": 30,
        "parquet_compressed_column_bytes": 40,
        "parquet_uncompressed_column_bytes": 100,
    }
    for entry in schedule["entries"]:
        codec = str(entry["parquet_compression"])
        output_storage = {
            "parquet_compression": codec,
            "registered_payload_bytes": (
                70 if codec == "zstd" else 110
            ),
            "manifest_bytes": 10,
            "tree_file_bytes": 80 if codec == "zstd" else 120,
            "parquet_file_bytes": 50 if codec == "zstd" else 90,
            "json_file_bytes": 30,
            "parquet_compressed_column_bytes": (
                40 if codec == "zstd" else 100
            ),
            "parquet_uncompressed_column_bytes": 100,
        }
        body = {
            "schema_version": (
                COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_OBSERVATION_SCHEMA
            ),
            "sequence_index": entry["sequence_index"],
            "observation_kind": entry["observation_kind"],
            "repetition_index": entry["repetition_index"],
            "parquet_compression": codec,
            "wall_seconds": 1.0 if codec == "zstd" else 2.0,
            "cpu_seconds": 0.5,
            "process_read_bytes": 100,
            "process_written_bytes": 50,
            "logical_byte_counters": (
                compression_module._logical_byte_counters(
                    source_storage=source_storage,
                    output_storage=output_storage,
                )
            ),
            "byte_accounting_basis": (
                "known_bulk_transcode_hash_reopen_and_semantic_parse_passes_v1"
            ),
            "output_storage": output_storage,
            "artifact_manifest_path": (
                f"/not-reopened/{codec}/cluster_preflight_manifest.json"
            ),
            "artifact_content_sha256": (
                "7" * 64 if codec == "zstd" else "8" * 64
            ),
            "payload_inventory_content_sha256": (
                "5" * 64 if codec == "zstd" else "6" * 64
            ),
            "path_neutral_scientific_content_sha256": scientific,
            "scientifically_equal_to_source": True,
            "status": "complete",
        }
        observations.append(
            {**body, "content_sha256": identity_sha256(body)}
        )
    warmups = [
        row
        for row in observations
        if row["observation_kind"] == "warmup"
    ]
    measured = [
        row
        for row in observations
        if row["observation_kind"] == "measured"
    ]
    codec_results, selected = compression_module._summaries(
        config=compression_config,
        source_scientific_sha256=scientific,
        observations=observations,
    )
    body = {
        "schema_version": (
            COMPACT_PREFLIGHT_COMPRESSION_BENCHMARK_RESULT_SCHEMA
        ),
        "status": "complete",
        "config": compression_config.as_dict(),
        "config_sha256": identity_sha256(
            compression_config.as_dict()
        ),
        "source": {
            "manifest_path": (
                "/not-reopened/source/cluster_preflight_manifest.json"
            ),
            "artifact_content_sha256": "4" * 64,
            "path_neutral_scientific_content_sha256": scientific,
            "logical_scope_count": 2,
            "physical_fit_count": 1,
            "physical_storage": {
                "owner_concept_payload_format": "parquet",
                "parquet_compression": "zstd",
                "parquet_use_dictionary": False,
                "parquet_write_statistics": False,
                "parquet_data_page_version": "1.0",
            },
            "storage": source_storage,
        },
        "execution_schedule": schedule,
        "warmup_observations": warmups,
        "warmup_observations_excluded_from_selection": True,
        "measured_observations": measured,
        "codec_results": codec_results,
        "selected_parquet_compression": selected,
        "selection_policy": (
            "lowest_median_wall_then_output_bytes_then_config_order_v1"
        ),
        "all_warmups_scientifically_equal": True,
        "cpu_only_serial_codec_benchmark": True,
        "accepted": True,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _bound_result(
    tmp_path: Path,
    *,
    accepted: bool = True,
) -> tuple[Path, Path, AuthenticatedPausedStage1Preflight]:
    config = RoleNeutralBenchmarkConfig.from_json(_BENCHMARK_CONFIG)
    request_sha256 = "a" * 64
    workflow_scientific_sha256 = "b" * 64
    preflight_sha256 = "c" * 64
    workflow_root = tmp_path / "staged-workflow"
    workflow_root.mkdir()
    prepared_root = tmp_path / "prepared-context"
    deployment = RoleNeutralBenchmarkWorkloadDeployment(
        workflow_root=workflow_root.resolve(),
        expected_workflow_request_sha256=request_sha256,
        prepared_context_root=prepared_root.resolve(),
        expected_benchmark_config_sha256=identity_sha256(config.as_dict()),
        representative_scope_selectors=tuple(
            RoleNeutralBenchmarkScopeSelector(
                scope_label=scope.label,
                logical_scope_kind=(
                    "full_outer"
                    if index == 0
                    else "exact_inner"
                ),
                ordinal=0,
            )
            for index, scope in enumerate(config.representative_scopes)
        ),
    )
    workload_path = tmp_path / "workload-deployment.json"
    workload_path.write_text(
        json.dumps(deployment.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    workload_sha256, _size = stable_file_sha256(workload_path)
    source = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
        "workflow_request_sha256": request_sha256,
        "workflow_scientific_sha256": workflow_scientific_sha256,
        "workload_deployment_sha256": workload_sha256,
        "stage1_preflight_phase_content_sha256": preflight_sha256,
        "prepared_stage1_context_content_root_sha256": "d" * 64,
    }
    workload_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
        "source": source,
        "representative_scope_plans": [
            {
                "scope_label": scope.label,
                "fit_row_count": scope.fit_row_count,
                "plan_scientific_content_sha256": f"{index + 1:x}" * 64,
                "physical_owner_scope_id": f"owner-{index}",
            }
            for index, scope in enumerate(config.representative_scopes)
        ],
    }
    workload_binding = {
        **workload_body,
        "content_sha256": identity_sha256(workload_body),
    }
    selected_name = "two_accelerators_two_fits_each"
    candidate_results: list[dict[str, Any]] = []
    for candidate in config.candidates:
        is_selected = candidate.name == selected_name
        candidate_results.append(
            {
                "candidate_name": candidate.name,
                "accepted": bool(accepted and is_selected),
                "executor_mode": candidate.executor_mode,
                "measured_observation_telemetry_accepted": bool(
                    accepted and is_selected
                ),
                "warmup_observation_telemetry_accepted": True,
                "warmup_scientific_identity_matches_measured": True,
                "execution_device_count": candidate.accelerator_count or 1,
                "concurrency_per_device": candidate.concurrency_per_device,
                "throughput_fit_rows_per_second": (
                    100.0 if is_selected else 1.0
                ),
                "scope_results": [
                    {
                        "scope_label": scope.label,
                        "deterministic_artifact_identity": True,
                        "scientific_artifact_sha256": (
                            f"{index + 1:x}" * 64
                        ),
                    }
                    for index, scope in enumerate(
                        config.representative_scopes
                    )
                ],
            }
        )
    schedule_entries = []
    sequence_index = 0
    repetitions = (
        config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
    )
    candidates = tuple(config.candidates)
    for scope_index, scope in enumerate(config.representative_scopes):
        for warmup_index in range(
            config.warmup_observations_per_candidate_scope
        ):
            rotation_offset = (
                scope_index
                * config.warmup_observations_per_candidate_scope
                + warmup_index
            ) % len(candidates)
            rotated = (
                candidates[rotation_offset:]
                + candidates[:rotation_offset]
            )
            for candidate_position, candidate in enumerate(rotated):
                schedule_entries.append(
                    {
                        "sequence_index": sequence_index,
                        "observation_kind": "warmup",
                        "scope_label": scope.label,
                        "observation_index": warmup_index,
                        "rotation_offset": rotation_offset,
                        "candidate_position": candidate_position,
                        "candidate_name": candidate.name,
                    }
                )
                sequence_index += 1
        for repetition_index in range(repetitions):
            rotation_offset = (
                scope_index * repetitions + repetition_index
            ) % len(candidates)
            rotated = (
                candidates[rotation_offset:]
                + candidates[:rotation_offset]
            )
            for candidate_position, candidate in enumerate(rotated):
                schedule_entries.append(
                    {
                        "sequence_index": sequence_index,
                        "observation_kind": "measured",
                        "scope_label": scope.label,
                        "observation_index": repetition_index,
                        "rotation_offset": rotation_offset,
                        "candidate_position": candidate_position,
                        "candidate_name": candidate.name,
                    }
                )
                sequence_index += 1
    schedule_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA,
        "warmup_policy": (
            "configured_complete_observations_excluded_from_selection_v1"
        ),
        "warmup_observations_per_candidate_scope": (
            config.warmup_observations_per_candidate_scope
        ),
        "candidate_order_policy": (
            "scope_observation_latin_rotation_with_warmup_v2"
        ),
        "candidate_names_in_configured_order": [
            candidate.name for candidate in candidates
        ],
        "entries": schedule_entries,
    }
    compression_result = _compression_result(config)
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
        "status": "complete",
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "workload_binding": workload_binding,
        "resource_inventory": {},
        "execution_schedule": {
            **schedule_body,
            "content_sha256": identity_sha256(schedule_body),
        },
        "warmup_observations": [],
        "warmup_telemetry": [],
        "warmup_observations_excluded_from_selection": True,
        "benchmark_observations": [],
        "observation_telemetry": [],
        "terminal_audit": {
            "exactly_one_completed_terminal_audit": True,
        },
        "terminal_audit_telemetry": {},
        "ordinary_observations_exclude_terminal_audit": True,
        "candidate_results": candidate_results,
        "preflight_compression_benchmark": compression_result,
        "benchmark_matrix_coverage": (
            build_role_neutral_benchmark_matrix_coverage(
                config=config,
                candidate_rows=candidate_results,
                compression_benchmark_result=compression_result,
            )
        ),
        "selected_candidate": selected_name,
        "selection_policy": (
            "fastest_end_to_end_then_lower_total_concurrency_then_name_v1"
        ),
        "scientific_result_identity_sha256": "e" * 64,
        "accepted": True,
    }
    value = {**body, "content_sha256": identity_sha256(body)}
    result_path = tmp_path / "benchmark-result.json"
    result_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    scientific = ScientificWorkflowSpec.from_json(_SCIENTIFIC)
    scientific_source_sha256, _size = stable_file_sha256(_SCIENTIFIC)
    authenticated = AuthenticatedPausedStage1Preflight(
        root=workflow_root.resolve(),
        request={
            "request_sha256": request_sha256,
            "portable_scientific_spec": scientific.identity_payload(),
            "scientific_spec_source_sha256": scientific_source_sha256,
            "scientific_identity": {
                "scientific_sha256": workflow_scientific_sha256,
            },
        },
        phases={
            "stage1_preflight": {
                "content_sha256": preflight_sha256,
            }
        },
    )
    return result_path, workload_path, authenticated


def _install_authenticated_source(
    monkeypatch: pytest.MonkeyPatch,
    authenticated: AuthenticatedPausedStage1Preflight,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_authenticate_paused_stage1_preflight",
        lambda _deployment, *, require_fresh_prepared_context: authenticated,
    )
    monkeypatch.setattr(
        selection_module,
        "_authenticate_prepared_context_binding",
        lambda **_kwargs: {
            "content_root_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        selection_module,
        "validate_compact_preflight_compression_benchmark_result",
        lambda value, *, reopen_artifacts: value,
    )


def _select(
    *,
    result: Path,
    workload: Path,
    output: Path,
) -> DeploymentProfile:
    return select_benchmarked_deployment_profile(
        base_deployment_path=_BASE_DEPLOYMENT,
        benchmark_result_path=result,
        benchmark_workload_deployment_path=workload,
        scientific_spec_path=_SCIENTIFIC,
        output_path=output,
    )


def test_measured_selection_publishes_fresh_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, workload, authenticated = _bound_result(tmp_path)
    _install_authenticated_source(monkeypatch, authenticated)
    output_parent = tmp_path / "different-output-parent"
    output_parent.mkdir()
    output = output_parent / "selected-deployment.json"
    scientific_before = ScientificWorkflowSpec.from_json(_SCIENTIFIC).scientific_sha256

    selected = _select(result=result, workload=workload, output=output)

    assert selected.devices == ("auto",)
    assert selected.stage1_execution.resource_kind == "accelerator"
    assert selected.stage1_execution.device_count == 2
    assert selected.stage1_execution.scope_workers_per_device == 2
    assert selected.cluster_preflight_parquet_compression == "zstd"
    assert (
        selected.stage1_execution.selection_method
        == "measured_role_neutral_benchmark_v1"
    )
    assert (
        selected.stage1_execution.selected_candidate
        == "two_accelerators_two_fits_each"
    )
    assert selected.stage1_execution.benchmark_result_sha256 == hashlib.sha256(
        result.read_bytes()
    ).hexdigest()
    assert selected.dataset_path == (
        _BASE_DEPLOYMENT.parent
        / "../synthetic_data/example_synthetic_datasets/"
        "one_confounder_one_effect_modifier_nsclc_with_structured/"
        "dataset.parquet"
    ).resolve()
    assert selected.dataset_path.is_absolute()
    assert DeploymentProfile.from_json(output) == selected
    assert output.stat().st_mode & 0o777 == 0o444
    validation = validate_benchmarked_stage1_execution_profile(
        profile=selected.stage1_execution,
        scientific_spec_path=_SCIENTIFIC,
        resource_performance_safety=(
            selected.resource_performance_safety
        ),
        cpu_budget=selected.cpu_budget,
    )
    assert validation["selected_candidate"] == (
        "two_accelerators_two_fits_each"
    )
    assert (
        validation["selected_preflight_parquet_compression"]
        == "zstd"
    )
    assert validation["benchmark_result_sha256"] == (
        selected.stage1_execution.benchmark_result_sha256
    )
    assert (
        validation["benchmark_matrix_coverage"][
            "all_required_axes_accounted"
        ]
        is True
    )
    assert len(
        validation["benchmark_matrix_coverage"]["axes"]
    ) == 6
    assert (
        validation["benchmark_workload_deployment_sha256"]
        == selected.stage1_execution.benchmark_workload_deployment_sha256
    )
    assert ScientificWorkflowSpec.from_json(_SCIENTIFIC).scientific_sha256 == scientific_before

    with pytest.raises(FileExistsError, match="fresh"):
        _select(result=result, workload=workload, output=output)


def test_unaccepted_or_tampered_benchmark_selection_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected, workload, authenticated = _bound_result(tmp_path, accepted=False)
    _install_authenticated_source(monkeypatch, authenticated)
    with pytest.raises(ValueError, match="selection policy|uniquely accepted"):
        _select(
            result=rejected,
            workload=workload,
            output=tmp_path / "rejected-output.json",
        )

    tamper_root = tmp_path / "tamper"
    tamper_root.mkdir()
    tampered, tampered_workload, authenticated = _bound_result(tamper_root)
    _install_authenticated_source(monkeypatch, authenticated)
    value = json.loads(tampered.read_text(encoding="utf-8"))
    value["config"]["candidates"][0]["concurrency_per_device"] = 2
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="content identity|config identity"):
        _select(
            result=tampered,
            workload=tampered_workload,
            output=tmp_path / "tampered-output.json",
        )


def test_workload_binding_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, workload, authenticated = _bound_result(tmp_path)
    _install_authenticated_source(monkeypatch, authenticated)
    value = json.loads(result.read_text(encoding="utf-8"))
    value["workload_binding"]["source"]["workflow_request_sha256"] = "f" * 64
    workload_body = {
        key: item
        for key, item in value["workload_binding"].items()
        if key != "content_sha256"
    }
    value["workload_binding"]["content_sha256"] = identity_sha256(workload_body)
    body = {
        key: item for key, item in value.items() if key != "content_sha256"
    }
    value["content_sha256"] = identity_sha256(body)
    result.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="staged workflow request"):
        _select(
            result=result,
            workload=workload,
            output=tmp_path / "output.json",
        )


def test_symlinked_benchmark_result_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, workload, authenticated = _bound_result(tmp_path)
    _install_authenticated_source(monkeypatch, authenticated)
    linked = tmp_path / "linked.json"
    linked.symlink_to(result)
    with pytest.raises(ValueError, match="private regular file"):
        _select(
            result=linked,
            workload=workload,
            output=tmp_path / "output.json",
        )
