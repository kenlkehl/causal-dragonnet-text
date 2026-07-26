from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

import oci.inference.role_neutral_performance_benchmark_publication as publication_module
from oci.inference.compact_preflight_compression_benchmark import (
    run_compact_preflight_compression_benchmark,
)
from oci.inference.performance_telemetry import BenchmarkRunObservation
from oci.inference.portable_workflow_spec import (
    ScientificWorkflowSpec,
    identity_sha256,
)
from oci.inference.production_text_preparation import stable_file_sha256
from oci.inference.production_stage1_role_neutral_execution import (
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA,
)
from oci.inference.role_neutral_performance_benchmark import (
    ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
    _benchmark_execution_schedule,
    build_role_neutral_benchmark_matrix_coverage,
)
from oci.inference.role_neutral_performance_benchmark_publication import (
    ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
    _inventory,
    publish_role_neutral_performance_benchmark,
    validate_role_neutral_performance_benchmark_publication,
)
from oci.inference.role_neutral_benchmark_workload_provider import (
    AuthenticatedPausedStage1Preflight,
    RoleNeutralBenchmarkScopeSelector,
    RoleNeutralBenchmarkWorkloadDeployment,
)
from tests.test_production_stage1_cluster_preflight_artifact_v2 import (
    _seal,
    portable_validators,
)
from tests.test_role_neutral_performance_benchmark import (
    _config,
    _inventory as benchmark_inventory,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_SCIENTIFIC = (
    _REPOSITORY
    / "example_configs"
    / "portable_all_evidence_scientific_nsclc.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)


def _address(body: dict) -> dict:
    return {**body, "content_sha256": identity_sha256(body)}


def _synthetic_completed_benchmark(
    tmp_path: Path,
) -> tuple[
    Path,
    dict,
    Path,
    AuthenticatedPausedStage1Preflight,
]:
    source = (tmp_path / "scratch_benchmark").resolve()
    source.mkdir()
    for name in (
        "warmups",
        "runs",
        "executor_sessions",
        "checkpoints",
        "interrupted_observations",
    ):
        (source / name).mkdir()

    preflight_parent = tmp_path / "preflight_source"
    preflight_parent.mkdir()
    _audit, _request, preflight = _seal(preflight_parent)
    config = _config(fit_row_count=640)
    compression = run_compact_preflight_compression_benchmark(
        config=config.preflight_compression_benchmark,
        source=preflight,
        output_root=source / "preflight_compression_benchmark",
    )
    schedule = _benchmark_execution_schedule(config)
    plan_sha256 = "6" * 64
    workflow_request_sha256 = "1" * 64
    scientific_spec = ScientificWorkflowSpec.from_json(_SCIENTIFIC)
    scientific_spec_source_sha256, _size = stable_file_sha256(
        _SCIENTIFIC
    )
    portable_scientific_spec = scientific_spec.identity_payload()
    configuration_body = {
        "schema_version": "fixture_scientific_configuration_v1",
        "scientific_settings": portable_scientific_spec,
        "dataset_content_sha256": "d" * 64,
    }
    scientific_configuration = {
        **configuration_body,
        "scientific_configuration_sha256": identity_sha256(
            configuration_body
        ),
    }
    phase_code_identities = {
        "input_preparation": "a" * 64,
        "stage1_preflight": "b" * 64,
        "stage1_modeling": "c" * 64,
    }
    workflow_producer_code_identity = identity_sha256(
        {
            "schema_version": (
                "workflow_phase_producer_code_aggregate_v1"
            ),
            "phase_producer_code_identities": phase_code_identities,
        }
    )
    scientific_identity_body = {
        "schema_version": "portable_all_evidence_scientific_identity_v3",
        "scientific_configuration_sha256": scientific_configuration[
            "scientific_configuration_sha256"
        ],
        "workflow_producer_code_identity": (
            workflow_producer_code_identity
        ),
        "phase_producer_code_identities": phase_code_identities,
    }
    scientific_identity = {
        **scientific_identity_body,
        "scientific_sha256": identity_sha256(
            scientific_identity_body
        ),
    }
    workflow_root = (tmp_path / "staged-workflow").resolve()
    workflow_root.mkdir()
    prepared_context_root = (tmp_path / "prepared-context").resolve()
    deployment = RoleNeutralBenchmarkWorkloadDeployment(
        workflow_root=workflow_root,
        expected_workflow_request_sha256=workflow_request_sha256,
        prepared_context_root=prepared_context_root,
        expected_benchmark_config_sha256=identity_sha256(
            config.as_dict()
        ),
        representative_scope_selectors=(
            RoleNeutralBenchmarkScopeSelector(
                scope_label="opaque-representative",
                logical_scope_kind="full_outer",
                ordinal=0,
            ),
        ),
    )
    workload_deployment_path = (
        tmp_path / "workload-deployment.json"
    ).resolve()
    _write_json(workload_deployment_path, deployment.as_dict())
    workload_deployment_sha256, _size = stable_file_sha256(
        workload_deployment_path
    )
    authenticated = AuthenticatedPausedStage1Preflight(
        root=workflow_root,
        request={
            "request_sha256": workflow_request_sha256,
            "portable_scientific_spec": portable_scientific_spec,
            "scientific_spec_source_sha256": (
                scientific_spec_source_sha256
            ),
            "scientific_configuration_identity": (
                scientific_configuration
            ),
            "phase_producer_code_identities": phase_code_identities,
            "workflow_producer_code_identity": (
                workflow_producer_code_identity
            ),
            "scientific_identity": scientific_identity,
        },
        phases={},
    )
    source_binding = {
        "workflow_request_sha256": workflow_request_sha256,
        "workflow_scientific_sha256": scientific_identity[
            "scientific_sha256"
        ],
        "workload_deployment_sha256": workload_deployment_sha256,
        "stage1_preflight_phase_content_sha256": "4" * 64,
        "prepared_stage1_context_content_root_sha256": "5" * 64,
        "schema_version": ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
    }
    workload_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
        "source": source_binding,
        "representative_scope_plans": [
            {
                "scope_label": "opaque-representative",
                "fit_row_count": 640,
                "plan_scientific_content_sha256": plan_sha256,
                "physical_owner_scope_id": "configured-owner",
            }
        ],
    }
    workload = _address(workload_body)
    compression_identity = preflight.identity()
    request_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA,
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "workload_binding": workload,
        "immutable_inputs_by_scope": [
            {
                "scope_label": "opaque-representative",
                "scientific_htr_training_batch_size": 4,
                "inputs": [
                    {
                        "content_sha256": "7" * 64,
                        "size_bytes": 100,
                    }
                ],
            }
        ],
        "compression_source": {
            "manifest_path": str(preflight.manifest_path),
            "artifact_content_sha256": compression_identity[
                "content_sha256"
            ],
            "path_neutral_scientific_content_sha256": (
                compression_identity[
                    "path_neutral_scientific_content_sha256"
                ]
            ),
        },
        "resource_resume_compatibility": {
            "cpu_count": 8,
            "accelerators": [],
            "content_sha256": "8" * 64,
        },
        "candidate_device_assignments": [
            {
                "candidate_name": candidate.name,
                "devices": [
                    f"cuda:{index}"
                    for index in range(candidate.accelerator_count)
                ]
                or ["cpu"],
            }
            for candidate in config.candidates
        ],
        "execution_schedule": schedule,
        "producer_code_evidence": {"fixture": "9" * 64},
    }
    request = _address(request_body)
    _write_json(source / "benchmark_request.json", request)

    scientific_sha256 = "a" * 64
    warmup_observations: list[dict] = []
    warmup_details: list[dict] = []
    measured_observations: list[dict] = []
    measured_details: list[dict] = []
    terminal_artifacts: list[dict] = []
    candidates_by_name = {
        candidate.name: candidate for candidate in config.candidates
    }
    for entry in schedule["entries"]:
        sequence_index = int(entry["sequence_index"])
        kind = str(entry["observation_kind"])
        observation_root = (
            source
            / ("warmups" if kind == "warmup" else "runs")
            / str(entry["candidate_name"])
            / str(entry["scope_label"])
            / (
                f"warmup_{int(entry['observation_index']):03d}"
                if kind == "warmup"
                else f"repetition_{int(entry['observation_index']):03d}"
            )
        )
        candidate = candidates_by_name[str(entry["candidate_name"])]
        complete_artifacts: list[dict] = []
        for fit_index in range(4):
            fit_root = observation_root / f"fit_{fit_index:03d}"
            fit_root.mkdir(parents=True)
            execution_body = {
                "schema_version": ROLE_NEUTRAL_STAGE1_EXECUTION_SCHEMA,
                "status": "complete",
                "plan_scientific_content_sha256": plan_sha256,
                "scientific_identity": {
                    "content_sha256": scientific_sha256
                },
                "every_physical_owner_executed_once": True,
                "every_component_executed_and_authenticated_once_per_owner": True,
                "coordination_gate_published_after_complete_execution": True,
            }
            execution = _address(execution_body)
            _write_json(
                fit_root / ROLE_NEUTRAL_EXECUTION_MANIFEST,
                execution,
            )
            fit_inventory, fit_tree, fit_bytes = _inventory(
                fit_root,
                require_read_only=True,
            )
            terminal_artifacts.append(
                {
                    "root": str(fit_root),
                    "tree_sha256": fit_tree,
                    "total_file_bytes": fit_bytes,
                    "file_count": len(fit_inventory),
                    "scientific_artifact_sha256": scientific_sha256,
                }
            )
            complete_artifacts.append(
                {
                    "relative_root": fit_root.relative_to(source).as_posix(),
                    "manifest_content_sha256": execution[
                        "content_sha256"
                    ],
                    "scientific_artifact_sha256": scientific_sha256,
                }
            )
        devices = tuple(
            f"cuda:{index}" for index in range(candidate.accelerator_count)
        ) or ("cpu",)
        observation = asdict(
            BenchmarkRunObservation(
                candidate_name=candidate.name,
                scope_label="opaque-representative",
                repetition_index=int(entry["observation_index"]),
                device_ids=devices,
                concurrency_per_device=candidate.concurrency_per_device,
                completed_scope_fits=4,
                model_fit_wall_seconds=1.0,
                peak_allocation_fraction=(
                    None if devices == ("cpu",) else 0.1
                ),
                minimum_observed_headroom_bytes=(
                    None if devices == ("cpu",) else 900
                ),
                oom_observed=False,
                scientific_artifact_sha256=scientific_sha256,
                artifact_path=str(observation_root),
                end_to_end_wall_seconds=1.1,
                complete_artifacts_exactly_equal=True,
            )
        )
        detail = {
            "candidate_name": candidate.name,
            "scope_label": "opaque-representative",
            "observation_kind": kind,
            "repetition_index": int(entry["observation_index"]),
            "execution_sequence_index": sequence_index,
            "candidate_position_within_rotation": int(
                entry["candidate_position"]
            ),
            "candidate_rotation_offset": int(entry["rotation_offset"]),
            "scientific_artifact_sha256": scientific_sha256,
            "complete_scientific_artifacts_exactly_equal": True,
            "telemetry_accepted": True,
        }
        observation_inventory, observation_tree, observation_bytes = (
            _inventory(observation_root, require_read_only=True)
        )
        checkpoint_body = {
            "schema_version": (
                ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA
            ),
            "request_sha256": request["content_sha256"],
            "schedule_entry": copy.deepcopy(entry),
            "observation": observation,
            "detail": detail,
            "observation_tree": {
                "tree_sha256": observation_tree,
                "total_file_bytes": observation_bytes,
                "file_count": len(observation_inventory),
            },
            "complete_artifacts": complete_artifacts,
        }
        _write_json(
            source
            / "checkpoints"
            / f"observation_{sequence_index:06d}.json",
            _address(checkpoint_body),
        )
        if kind == "warmup":
            warmup_observations.append(observation)
            warmup_details.append(detail)
        else:
            measured_observations.append(observation)
            measured_details.append(detail)

    candidate_rows = [
        {
            "candidate_name": candidate.name,
            "scope_results": [
                {
                    "scope_label": "opaque-representative",
                    "deterministic_artifact_identity": True,
                    "scientific_artifact_sha256": scientific_sha256,
                }
            ],
            "warmup_observation_telemetry_accepted": True,
            "measured_observation_telemetry_accepted": True,
            "warmup_scientific_identity_matches_measured": True,
            "htr_operational_attestations_accepted": True,
            "cpu_gpu_lane_interval_telemetry_accepted": True,
            "cpu_gpu_lane_overlap_observation_count": (
                1 if candidate.accelerator_count > 0 else 0
            ),
            "cpu_gpu_lane_overlap_descriptive_only": True,
            "cpu_gpu_lane_overlap_speedup_claimed": False,
            "executor_mode": candidate.executor_mode,
            "execution_device_count": candidate.accelerator_count or 1,
            "concurrency_per_device": candidate.concurrency_per_device,
            "resource_slot_count": candidate.resource_slot_count,
            "effective_parallel_owners": candidate.total_concurrency,
            "neural_query_topology": (
                candidate.neural_query_topology.as_dict()
            ),
            "htr_operational_controls": (
                candidate.htr_operational_controls.as_dict()
            ),
            "throughput_fit_rows_per_second": (
                100.0
                if candidate.name == config.scientific_reference_candidate
                else 1.0
            ),
            "accepted": True,
        }
        for candidate in config.candidates
    ]
    matrix = build_role_neutral_benchmark_matrix_coverage(
        config=config,
        candidate_rows=candidate_rows,
        compression_benchmark_result=compression,
    )
    result_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
        "status": "complete",
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "workload_binding": workload,
        "resource_inventory": benchmark_inventory().as_dict(),
        "execution_schedule": schedule,
        "warmup_observations": warmup_observations,
        "warmup_telemetry": warmup_details,
        "warmup_observations_excluded_from_selection": True,
        "benchmark_observations": measured_observations,
        "observation_telemetry": measured_details,
        "terminal_audit": {
            "exactly_one_completed_terminal_audit": True,
            "audited_complete_artifact_count": len(terminal_artifacts),
            "artifacts": terminal_artifacts,
        },
        "terminal_audit_telemetry": {
            "subphases": [{"status": "completed"}]
        },
        "ordinary_observations_exclude_terminal_audit": True,
        "candidate_results": candidate_rows,
        "preflight_compression_benchmark": compression,
        "benchmark_matrix_coverage": matrix,
        "selected_candidate": config.scientific_reference_candidate,
        "selection_policy": (
            "fastest_end_to_end_then_lower_effective_owner_concurrency_then_name_v2"
        ),
        "scientific_result_identity_sha256": "b" * 64,
        "accepted": True,
    }
    result = _address(result_body)
    _write_json(source / "benchmark_result.json", result)
    return source, result, workload_deployment_path, authenticated


def test_compact_durable_publication_is_path_neutral_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    portable_validators,
) -> None:
    source, result, workload_deployment, authenticated = (
        _synthetic_completed_benchmark(tmp_path)
    )
    monkeypatch.setattr(
        publication_module,
        "_authenticate_paused_stage1_preflight",
        lambda _deployment, *, require_fresh_prepared_context: authenticated,
    )
    durable_a = (tmp_path / "durable-a").resolve()
    durable_b = (tmp_path / "durable-b").resolve()
    first = publish_role_neutral_performance_benchmark(
        scratch_root=source,
        durable_root=durable_a,
        workload_deployment_path=workload_deployment,
    )
    second = publish_role_neutral_performance_benchmark(
        scratch_root=source,
        durable_root=durable_b,
        workload_deployment_path=workload_deployment,
    )

    assert first.path_neutral_content_root_sha256 == (
        second.path_neutral_content_root_sha256
    )
    assert first.source_record_policy[
        "historical_scratch_locators_authoritative"
    ] is False
    assert first.source_record_policy[
        "complete_fit_replica_trees_published"
    ] is False
    assert first.canonical_scientific_artifact[
        "raw_artifact_tree_published"
    ] is False
    assert not (durable_a / "runs").exists()
    assert {
        path.name
        for path in (durable_a / "canonical_scientific_artifact").iterdir()
    } == {"execution_manifest.json", "artifact_reference.json"}
    normalized = (
        durable_a
        / "logical_evidence"
        / "path_neutral_benchmark_result.json"
    ).read_text(encoding="utf-8")
    assert str(source) not in normalized
    assert str(durable_a) not in normalized
    assert str(durable_b) not in normalized
    assert validate_role_neutral_performance_benchmark_publication(
        durable_a
    ).content_sha256 == first.content_sha256
    manifest_mtime = os.lstat(
        durable_a / ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST
    ).st_mtime_ns
    assert all(
        os.lstat(durable_a / row.relative_path).st_mtime_ns
        <= manifest_mtime
        for row in first.payload_inventory
    )

    target = durable_a / "logical_evidence" / "benchmark_config.json"
    original = target.read_bytes()
    target.chmod(0o644)
    target.write_bytes(original + b" ")
    target.chmod(0o444)
    with pytest.raises(ValueError, match="payload bytes changed"):
        validate_role_neutral_performance_benchmark_publication(durable_a)
    target.chmod(0o644)
    target.write_bytes(original)
    target.chmod(0o444)
    validate_role_neutral_performance_benchmark_publication(durable_a)

    source_records = durable_a / "source_records"
    source_records.chmod(0o755)
    linked = source_records / "hard-linked-result.json"
    os.link(source_records / "benchmark_result.json", linked)
    source_records.chmod(0o555)
    with pytest.raises(ValueError, match="private read-only"):
        validate_role_neutral_performance_benchmark_publication(durable_a)
    source_records.chmod(0o755)
    linked.unlink()
    source_records.chmod(0o555)

    source_records.chmod(0o755)
    symlink = source_records / "linked-request.json"
    symlink.symlink_to("benchmark_request.json")
    source_records.chmod(0o555)
    with pytest.raises(ValueError, match="symlink"):
        validate_role_neutral_performance_benchmark_publication(durable_a)
    source_records.chmod(0o755)
    symlink.unlink()
    source_records.chmod(0o555)

    durable_a.chmod(0o755)
    extra = durable_a / "unrelated.json"
    extra.write_text("{}\n", encoding="utf-8")
    extra.chmod(0o444)
    durable_a.chmod(0o555)
    with pytest.raises(ValueError, match="extra/missing"):
        validate_role_neutral_performance_benchmark_publication(durable_a)
    durable_a.chmod(0o755)
    extra.unlink()
    durable_a.chmod(0o555)

    first_fit_manifest = next(
        (source / "runs").rglob(ROLE_NEUTRAL_EXECUTION_MANIFEST)
    )
    original_manifest = first_fit_manifest.read_bytes()
    first_fit_manifest.chmod(0o644)
    first_fit_manifest.write_bytes(original_manifest + b" ")
    first_fit_manifest.chmod(0o444)
    rejected_root = (tmp_path / "rejected-publication").resolve()
    with pytest.raises(ValueError, match="observation tree changed"):
        publish_role_neutral_performance_benchmark(
            scratch_root=source,
            durable_root=rejected_root,
            workload_deployment_path=workload_deployment,
        )
    assert not rejected_root.exists()
    assert result["accepted"] is True
