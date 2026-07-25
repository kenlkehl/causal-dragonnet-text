from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from oci.inference import performance_telemetry
from oci.inference import production_stage1_scope_scheduler as scheduler
from oci.inference import role_neutral_performance_benchmark as benchmark_module
from oci.inference.compact_preflight_compression_benchmark import (
    CompactPreflightCompressionBenchmarkConfig,
)
from oci.inference.performance_telemetry import ImmutableInputObservation
from oci.inference.portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
)
from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
)
from oci.inference.production_stage1_role_neutral_execution import (
    LocalThreadRoleNeutralPhysicalOwnerExecutor,
)
from oci.inference.role_neutral_performance_benchmark import (
    ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA,
    RoleNeutralBenchmarkCandidate,
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkScope,
    RoleNeutralBenchmarkSourceBinding,
    RoleNeutralBenchmarkWorkload,
    run_role_neutral_performance_benchmark,
)
from tests import test_production_stage1_role_neutral_execution as execution_support
from tests.test_production_stage1_cluster_preflight_artifact_v2 import (
    _seal,
    portable_validators,
)


def _one_owner_plan():
    source = execution_support._plan(gpu_ids=())
    owner, members = source.physical_scope_groups[0]
    member_ids = {value.scope_id for value in members}
    assignments = tuple(value for value in source.assignments if value.scope_id in member_ids)
    body = scheduler._stage1_scope_plan_body(
        registry_content_sha256=source.registry_content_sha256,
        global_seed=source.global_seed,
        review_rounds=source.review_rounds,
        initial_training_partitions=source.initial_training_partitions,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=members,
        assignments=assignments,
    )
    return scheduler.Stage1ScopePlan(
        registry_content_sha256=source.registry_content_sha256,
        global_seed=source.global_seed,
        review_rounds=source.review_rounds,
        initial_training_partitions=source.initial_training_partitions,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=members,
        assignments=assignments,
        content_sha256=scheduler._sha256_json(body),
    )


def _config(*, fit_row_count: int) -> RoleNeutralBenchmarkConfig:
    safety = ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=0.9,
        gpu_minimum_headroom_bytes=100,
        minimum_multi_device_throughput_ratio=0.01,
        maximum_coordination_proof_overhead_ratio=1_000_000.0,
        maximum_ordinary_read_amplification=1_000_000.0,
        minimum_benchmark_repetitions_per_scope=2,
        read_counter_source="process_read_bytes",
        fail_on_external_gpu_occupants=True,
    )
    return RoleNeutralBenchmarkConfig(
        representative_scopes=(
            RoleNeutralBenchmarkScope(
                label="opaque-representative",
                fit_row_count=fit_row_count,
                fits_per_observation=4,
            ),
        ),
        candidates=(
            RoleNeutralBenchmarkCandidate(
                name="configured-single",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="fresh_per_fit",
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-single-persistent",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-single-two",
                accelerator_count=1,
                concurrency_per_device=2,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-multi",
                accelerator_count=2,
                concurrency_per_device=2,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
            ),
        ),
        scientific_reference_candidate="configured-single",
        multi_device_baselines=(
            ("configured-multi", "configured-single-persistent"),
        ),
        resource_performance_safety=safety,
        preflight_compression_benchmark=(
            CompactPreflightCompressionBenchmarkConfig(
                codecs=("none", "zstd"),
                warmup_repetitions_per_codec=0,
                measured_repetitions_per_codec=1,
            )
        ),
        gpu_sample_interval_seconds=0.01,
        warmup_observations_per_candidate_scope=1,
    )


def _inventory() -> ResourceInventory:
    return ResourceInventory(
        cpu_count=8,
        gpus=(
            GPUResource(
                device="cuda:4",
                uuid="configured-a",
                total_memory_bytes=1_000,
                free_memory_bytes=900,
                utilization_percent=0.0,
            ),
            GPUResource(
                device="cuda:9",
                uuid="configured-b",
                total_memory_bytes=1_000,
                free_memory_bytes=900,
                utilization_percent=0.0,
            ),
        ),
    )


def test_real_role_neutral_callbacks_are_repeated_measured_and_audited_once(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(fit_row_count=plan.physical_scopes[0].fit_row_count)
    _audit, _request, preflight_source = _seal(tmp_path)

    def samples(devices):
        return tuple(
            {
                "device": device,
                "uuid": f"uuid-{device}",
                "utilization_percent": 37.0,
                "memory_used_bytes": 100,
                "memory_total_bytes": 1_000,
            }
            for device in devices
            if device.startswith("cuda:")
        )

    monkeypatch.setattr(performance_telemetry, "sample_nvidia_gpus", samples)
    monkeypatch.setattr(
        performance_telemetry,
        "_reset_torch_peaks",
        lambda _devices: None,
    )
    monkeypatch.setattr(
        performance_telemetry,
        "_torch_peaks",
        lambda devices: {device: 100 for device in devices},
    )
    observation_io = iter(
        value
        for observation in range(12)
        for value in (
            (observation * 1_000, observation * 2_000),
            (observation * 1_000 + 137, observation * 2_000 + 211),
        )
    )
    monkeypatch.setattr(
        benchmark_module,
        "_observation_process_io",
        lambda: next(observation_io),
    )
    persistent_sessions = []

    class LocalPersistentSession:
        def __init__(self) -> None:
            self.delegate = LocalThreadRoleNeutralPhysicalOwnerExecutor()
            self.execute_calls = 0
            self.closed = False

        def execute(self, **kwargs):
            self.execute_calls += 1
            return self.delegate.execute(**kwargs)

        def close(self) -> None:
            self.closed = True

    class LocalPersistentBase:
        def open_session(self, **_kwargs):
            session = LocalPersistentSession()
            persistent_sessions.append(session)
            return session

    def build_executor(mode: str, _workers: int):
        if mode == "persistent_slots":
            return LocalPersistentBase()
        return LocalThreadRoleNeutralPhysicalOwnerExecutor()

    workload = RoleNeutralBenchmarkWorkload(
        scope_label="opaque-representative",
        plan=plan,
        producer_factories_builder=lambda: (execution_support._ProducerRecorder().factories()),
        physical_owner_executor_builder=build_executor,
        preflight_compression_source_builder=lambda: preflight_source,
        immutable_inputs=(
            ImmutableInputObservation(
                content_sha256=hashlib.sha256(b"fixture-input").hexdigest(),
                size_bytes=1_000_000,
            ),
        ),
        source_binding=RoleNeutralBenchmarkSourceBinding(
            workflow_request_sha256="1" * 64,
            workflow_scientific_sha256="2" * 64,
            workload_deployment_sha256="3" * 64,
            stage1_preflight_phase_content_sha256="4" * 64,
            prepared_stage1_context_content_root_sha256="5" * 64,
        ),
    )
    result = run_role_neutral_performance_benchmark(
        config=config,
        workloads={workload.scope_label: workload},
        output_root=(tmp_path / "benchmark").resolve(),
        inventory=_inventory(),
    )

    assert result["accepted"] is True
    assert len(persistent_sessions) == 3
    assert {
        session.execute_calls for session in persistent_sessions
    } == {12}
    assert all(session.closed for session in persistent_sessions)
    assert result["selected_candidate"] in {
        "configured-single",
        "configured-single-persistent",
        "configured-multi",
    }
    assert len(result["benchmark_observations"]) == 8
    assert {row["completed_scope_fits"] for row in result["benchmark_observations"]} == {4}
    assert all(row["complete_artifacts_exactly_equal"] for row in result["benchmark_observations"])
    assert all(row["end_to_end_wall_seconds"] > 0 for row in result["benchmark_observations"])
    assert all(
        row["peak_allocation_fraction"]
        < config.resource_performance_safety.gpu_max_allocation_fraction
        for row in result["benchmark_observations"]
    )
    assert all(
        row["minimum_observed_headroom_bytes"]
        >= config.resource_performance_safety.gpu_minimum_headroom_bytes
        for row in result["benchmark_observations"]
    )
    assert all(
        {phase["activity_kind"] for phase in row["phase_telemetry"]}
        == {"model_fit", "coordination_proof"}
        for row in result["observation_telemetry"]
    )
    terminal = result["terminal_audit_telemetry"]["subphases"]
    assert len(terminal) == 1
    assert terminal[0]["activity_kind"] == "terminal_audit"
    assert terminal[0]["byte_counters"]["read"] > 0
    assert result["ordinary_observations_exclude_terminal_audit"] is True
    assert result["workload_binding"]["source"] == workload.source_binding.as_dict()
    schedule = result["execution_schedule"]
    assert schedule["warmup_policy"] == (
        "configured_complete_observations_excluded_from_selection_v1"
    )
    assert schedule["candidate_order_policy"] == (
        "scope_observation_latin_rotation_with_warmup_v2"
    )
    assert [
        row["candidate_name"]
        for row in schedule["entries"]
        if row["candidate_position"] == 0
    ] == [
        "configured-single",
        "configured-single",
        "configured-single-persistent",
    ]
    assert [
        row["execution_sequence_index"]
        for row in result["observation_telemetry"]
    ] == list(range(4, 12))
    assert [
        row["execution_sequence_index"]
        for row in result["warmup_telemetry"]
    ] == [0, 1, 2, 3]
    assert result["warmup_observations_excluded_from_selection"] is True
    matrix = result["benchmark_matrix_coverage"]
    assert matrix["all_required_axes_accounted"] is True
    assert len(matrix["axes"]) == 6
    assert matrix["axes"][0]["disposition"] == "measured"
    assert matrix["axes"][1]["disposition"] == "measured"
    assert matrix["axes"][2]["disposition"] == (
        "scientific_configuration_not_operationally_tunable"
    )
    assert matrix["axes"][5]["disposition"] == "partially_measured"
    assert (
        matrix["axes"][5]["component_dispositions"][0]["disposition"]
        == "measured"
    )
    assert (
        matrix["axes"][5]["component_dispositions"][1]["disposition"]
        == "unsupported_by_v1_executor"
    )
    without_single_device_two_fit = replace(
        config,
        candidates=tuple(
            candidate
            for candidate in config.candidates
            if candidate.name != "configured-single-two"
        ),
    )
    no_inference = (
        benchmark_module.build_role_neutral_benchmark_matrix_coverage(
            config=without_single_device_two_fit,
            candidate_rows=[
                row
                for row in result["candidate_results"]
                if row["candidate_name"] != "configured-single-two"
            ],
            compression_benchmark_result=result[
                "preflight_compression_benchmark"
            ],
        )
    )
    assert (
        no_inference["axes"][1]["disposition"]
        == "unsupported_by_v1_executor"
    )
    assert no_inference["axes"][1]["performance_claimed"] is False
    assert all(
        row["performance_claimed"] is False
        for row in matrix["axes"][2:]
    )
    assert {
        (
            row["candidate_name"],
            row["host_cpu_budget"],
            row["per_fit_cpu_budget"],
            row["maximum_simultaneous_fit_cpu_budget"],
        )
        for row in result["observation_telemetry"]
    } == {
        ("configured-single", 4, 4, 4),
        ("configured-single-persistent", 4, 4, 4),
        ("configured-single-two", 4, 2, 4),
        ("configured-multi", 4, 1, 4),
    }
    assert all(
        row["terminal_audit_read_bytes_included"] is False
        for row in result["observation_telemetry"]
    )
    assert {
        row["ordinary_read_bytes"] for row in result["observation_telemetry"]
    } == {137}
    assert {
        row["observation_parent_process_written_bytes"]
        for row in result["observation_telemetry"]
    } == {211}
    assert {
        row["process_counter_attribution"]
        for row in result["observation_telemetry"]
    } == {"one_parent_process_delta_per_complete_observation"}


def test_checked_nsclc_matrix_is_typed_and_sizes_are_not_in_runner_source() -> None:
    path = (
        Path(__file__).parents[1]
        / "example_configs"
        / "portable_role_neutral_performance_benchmark_nsclc.deployment.json"
    )
    config = RoleNeutralBenchmarkConfig.from_json(path)
    assert config.schema_version == ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA
    assert len(config.representative_scopes) == 2
    assert len(config.candidates) == 5
    assert {value.executor_mode for value in config.candidates} == {
        "fresh_per_fit",
        "persistent_slots",
    }
    assert all(value.accelerator_count > 0 for value in config.candidates)
    assert {
        value.concurrency_per_device
        for value in config.candidates
        if value.accelerator_count == 1
        and value.executor_mode == "persistent_slots"
    } == {1, 2}
    assert all(
        scope.fits_per_observation
        >= max(candidate.total_concurrency for candidate in config.candidates)
        for scope in config.representative_scopes
    )
    source_paths = (
        Path(__file__).parents[1] / "oci" / "inference" / "role_neutral_performance_benchmark.py",
        Path(__file__).parents[1] / "scripts" / "run_role_neutral_performance_benchmark.py",
        Path(__file__).parents[1]
        / "oci"
        / "inference"
        / "role_neutral_benchmark_workload_provider.py",
        Path(__file__).parents[1]
        / "scripts"
        / "write_role_neutral_benchmark_workload_deployment.py",
    )
    configured_sizes = {str(scope.fit_row_count) for scope in config.representative_scopes}
    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        assert not any(value in source for value in configured_sizes)
        assert "cuda:0" not in source
        assert "cuda:1" not in source


def test_process_isolated_fit_counters_are_closed_and_attributed_once() -> None:
    manifest = {
        "owner_execution_telemetry": {
            "process_isolated_physical_owners": True,
            "parent_process_counters_included_in_child_counters": False,
            "physical_owners": [
                {
                    "telemetry": {
                        "schema_version": (
                            "production_role_neutral_process_owner_telemetry_v1"
                        ),
                        "wall_seconds": 1.25,
                        "cpu_seconds": 0.75,
                        "peak_gpu_allocated_bytes": 505,
                        "peak_gpu_reserved_bytes": 606,
                        "process_io_deltas": {
                            "rchar": 101,
                            "wchar": 202,
                            "read_bytes": 303,
                            "write_bytes": 404,
                        },
                    }
                }
            ],
        }
    }
    assert benchmark_module._child_process_io_from_manifest(manifest) == (
        True,
        303,
        404,
        1.25,
        0.75,
        606,
    )
    manifest["owner_execution_telemetry"]["physical_owners"][0][
        "telemetry"
    ]["process_io_deltas"].pop("rchar")
    with pytest.raises(ValueError, match="closed child I/O"):
        benchmark_module._child_process_io_from_manifest(manifest)


@pytest.mark.parametrize(
    "unsafe",
    ("../escape", "nested/name", "/absolute", ".", "..", " \t../escape "),
)
def test_benchmark_names_cannot_escape_the_fresh_output_root(unsafe: str) -> None:
    with pytest.raises(ValueError, match="traversal-safe"):
        RoleNeutralBenchmarkCandidate(
            name=unsafe,
            accelerator_count=1,
            concurrency_per_device=1,
            host_cpu_budget=1,
            executor_mode="fresh_per_fit",
        )
    with pytest.raises(ValueError, match="traversal-safe"):
        RoleNeutralBenchmarkScope(
            label=unsafe,
            fit_row_count=1,
            fits_per_observation=1,
        )
