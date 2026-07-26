from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.resource_safety_test_support import resource_safety_policy

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
from oci.inference.role_neutral_htr_group_execution import (
    RoleNeutralHTROperationalControls,
)
from oci.inference.stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
    ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    Stage1ExecutionTopologyPolicy,
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
        physical_fit_identity=source.physical_fit_identity,
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
        physical_fit_identity=source.physical_fit_identity,
        gpu_ids=(),
        scope_workers_per_gpu=1,
        scopes=members,
        assignments=assignments,
        content_sha256=scheduler._sha256_json(body),
    )


def _single_device_topology() -> Stage1ExecutionTopologyPolicy:
    return Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_PER_SELECTED_DEVICE,
    )


def _spanning_topology() -> Stage1ExecutionTopologyPolicy:
    return Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    )


def _config(*, fit_row_count: int) -> RoleNeutralBenchmarkConfig:
    safety = resource_safety_policy(
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
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-single-persistent",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-single-two",
                accelerator_count=1,
                concurrency_per_device=2,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-multi",
                accelerator_count=2,
                concurrency_per_device=2,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-multi-span",
                accelerator_count=2,
                concurrency_per_device=2,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_spanning_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-htr-encoder",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=16,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=False,
                    chunk_plan_cache_max_entries=0,
                    tokenized_chunk_cache_max_entries=0,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-htr-workers",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=2,
                    reuse_tokenizer_and_chunk_plans=True,
                    chunk_plan_cache_max_entries=100,
                    tokenized_chunk_cache_max_entries=1000,
                ),
            ),
            RoleNeutralBenchmarkCandidate(
                name="configured-htr-reuse",
                accelerator_count=1,
                concurrency_per_device=1,
                host_cpu_budget=4,
                executor_mode="persistent_slots",
                neural_query_topology=_single_device_topology(),
                htr_operational_controls=RoleNeutralHTROperationalControls(
                    training_batch_size=4,
                    sentence_encoder_batch_size=8,
                    data_loader_workers=0,
                    reuse_tokenizer_and_chunk_plans=True,
                    chunk_plan_cache_max_entries=100,
                    tokenized_chunk_cache_max_entries=1000,
                ),
            ),
        ),
        scientific_reference_candidate="configured-single",
        multi_device_baselines=(
            ("configured-multi", "configured-single-two"),
            ("configured-multi-span", "configured-single-two"),
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


def _patch_measurement_telemetry(monkeypatch) -> None:
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
    state = {"call": 0}

    def process_io():
        call = state["call"]
        state["call"] += 1
        observation = call // 2
        completed = call % 2
        return (
            observation * 1_000 + (137 if completed else 0),
            observation * 2_000 + (211 if completed else 0),
        )

    monkeypatch.setattr(
        benchmark_module,
        "_observation_process_io",
        process_io,
    )


def _tracked_workload(
    *,
    plan,
    preflight_source,
    sessions: list,
    live_sessions: dict[str, int],
) -> RoleNeutralBenchmarkWorkload:
    class LocalPersistentSession:
        def __init__(self) -> None:
            self.delegate = LocalThreadRoleNeutralPhysicalOwnerExecutor()
            self.execute_calls = 0
            self.closed = False

        def execute(self, **kwargs):
            self.execute_calls += 1
            return _execute_test_tasks_allowing_spanning(
                delegate=self.delegate,
                **kwargs,
            )

        def close(self) -> None:
            assert self.closed is False
            self.closed = True
            live_sessions["current"] -= 1

    class LocalPersistentBase:
        def open_session(self, **_kwargs):
            assert live_sessions["current"] == 0
            live_sessions["current"] += 1
            live_sessions["maximum"] = max(
                live_sessions["maximum"],
                live_sessions["current"],
            )
            session = LocalPersistentSession()
            sessions.append(session)
            return session

    def build_executor(mode: str, _workers: int):
        if mode == "persistent_slots":
            return LocalPersistentBase()
        return LocalThreadRoleNeutralPhysicalOwnerExecutor()

    return RoleNeutralBenchmarkWorkload(
        scope_label="opaque-representative",
        plan=plan,
        scientific_htr_training_batch_size=4,
        producer_factories_builder=lambda: (
            execution_support._ProducerRecorder().factories()
        ),
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


def _execute_test_tasks_allowing_spanning(*, delegate, **kwargs):
    """Exercise test callbacks without weakening the production executor.

    The local production thread executor intentionally cannot atomically
    reserve several devices.  These callback tests have no GPU runtime, so a
    spanning task is invoked directly and remains operationally unattested.
    """

    tasks = tuple(kwargs["tasks"])
    if any(
        task.neural_query_execution_topology is not None
        and task.neural_query_execution_topology.spans_multiple_devices
        for task in tasks
    ):
        assert int(kwargs["max_workers"]) == 1
        assert int(kwargs["cpu_budget"]) >= 1
        worker = kwargs["worker"]
        return tuple(worker(task) for task in tasks)
    return delegate.execute(**kwargs)


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
    configured_observation_count = (
        len(config.representative_scopes)
        * len(config.candidates)
        * (
            config.warmup_observations_per_candidate_scope
            + config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
        )
    )
    observation_io = iter(
        value
        for observation in range(configured_observation_count)
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
    live_sessions = {"current": 0, "maximum": 0}

    class LocalPersistentSession:
        def __init__(self) -> None:
            self.delegate = LocalThreadRoleNeutralPhysicalOwnerExecutor()
            self.execute_calls = 0
            self.closed = False

        def execute(self, **kwargs):
            self.execute_calls += 1
            return _execute_test_tasks_allowing_spanning(
                delegate=self.delegate,
                **kwargs,
            )

        def close(self) -> None:
            assert self.closed is False
            self.closed = True
            live_sessions["current"] -= 1

    class LocalPersistentBase:
        def open_session(self, **_kwargs):
            assert live_sessions["current"] == 0
            live_sessions["current"] += 1
            live_sessions["maximum"] = max(
                live_sessions["maximum"],
                live_sessions["current"],
            )
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
        scientific_htr_training_batch_size=4,
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
    observations_per_candidate = (
        config.warmup_observations_per_candidate_scope
        + config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
    )
    assert len(persistent_sessions) == (
        observations_per_candidate
        * sum(
            candidate.executor_mode == "persistent_slots"
            for candidate in config.candidates
        )
    )
    assert {
        session.execute_calls for session in persistent_sessions
    } == {4}
    assert all(session.closed for session in persistent_sessions)
    assert live_sessions == {"current": 0, "maximum": 1}
    assert result["selected_candidate"] in {
        candidate.name for candidate in config.candidates
    }
    assert len(result["benchmark_observations"]) == (
        len(config.candidates)
        * config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
    )
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
    ] == list(
        range(
            len(config.candidates)
            * config.warmup_observations_per_candidate_scope,
            len(config.candidates)
            * observations_per_candidate,
        )
    )
    assert [
        row["execution_sequence_index"]
        for row in result["warmup_telemetry"]
    ] == list(
        range(
            len(config.candidates)
            * config.warmup_observations_per_candidate_scope
        )
    )
    assert result["warmup_observations_excluded_from_selection"] is True
    matrix = result["benchmark_matrix_coverage"]
    assert matrix["all_required_axes_accounted"] is True
    assert len(matrix["axes"]) == 6
    assert matrix["axes"][0]["disposition"] == "measured"
    assert matrix["axes"][1]["disposition"] == "measured"
    assert matrix["axes"][2]["disposition"] == "partially_measured"
    assert matrix["axes"][2]["component_dispositions"][0]["disposition"] == (
        "scientific_configuration_not_operationally_tunable"
    )
    assert all(
        row["disposition"] == "measured"
        and row["matched_one_factor_pairs"]
        for row in matrix["axes"][2]["component_dispositions"][1:]
    )
    assert matrix["axes"][3]["disposition"] == "measured"
    assert matrix["axes"][3]["matched_one_factor_pairs"]
    assert matrix["axes"][5]["disposition"] == "partially_measured"
    assert (
        matrix["axes"][5]["component_dispositions"][0]["disposition"]
        == "measured"
    )
    assert (
        matrix["axes"][5]["component_dispositions"][1]["disposition"]
        == "descriptively_measured"
    )
    lane_component = matrix["axes"][5]["component_dispositions"][1]
    assert lane_component["performance_claimed"] is False
    assert lane_component["causal_speedup_claimed"] is False
    assert lane_component["throughput_speedup_estimated"] is False
    accelerator_rows = [
        row
        for row in result["observation_telemetry"]
        if row["cpu_gpu_lane_interval_telemetry_required"]
    ]
    assert accelerator_rows
    assert all(
        row["cpu_gpu_lane_interval_telemetry_complete"] is True
        and row["cpu_gpu_lane_overlap_descriptive_only"] is True
        and row["cpu_gpu_lane_overlap_speedup_claimed"] is False
        for row in accelerator_rows
    )
    for row in accelerator_rows:
        closed = (
            benchmark_module.CompletedFitIntervalObservation.from_mapping(
                row["cpu_gpu_lane_interval_observation"]
            )
        )
        assert len(closed.intervals) == (
            row["configured_fits_per_observation"]
            * len(execution_support.EXPECTED_COMPONENT_FAMILIES)
        )
        assert all(
            closed.observation_started_monotonic_ns
            <= interval.started_monotonic_ns
            < interval.finished_monotonic_ns
            <= closed.observation_finished_monotonic_ns
            for interval in closed.intervals
        )
        recomputed = (
            benchmark_module.analyze_completed_fit_lane_overlap(
                closed,
                expected_observation_id=closed.observation_id,
                expected_owner_execution_ids=(
                    closed.owner_execution_ids
                ),
            )
        )
        assert recomputed.as_dict() == row[
            "cpu_gpu_lane_overlap_analysis"
        ]
        assert recomputed.causal_speedup_claimed is False
        assert recomputed.throughput_speedup_estimated is False
    without_single_device_two_fit = replace(
        config,
        candidates=tuple(
            candidate
            for candidate in config.candidates
            if candidate.name
            not in {
                "configured-single-two",
                "configured-multi",
                "configured-multi-span",
            }
        ),
        multi_device_baselines=(),
    )
    no_inference = (
        benchmark_module.build_role_neutral_benchmark_matrix_coverage(
            config=without_single_device_two_fit,
            candidate_rows=[
                row
                for row in result["candidate_results"]
                if row["candidate_name"]
                not in {
                    "configured-single-two",
                    "configured-multi",
                    "configured-multi-span",
                }
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
    assert {
        (
            row["candidate_name"],
            row["host_cpu_budget"],
            row["per_fit_cpu_budget"],
            row["maximum_simultaneous_fit_cpu_budget"],
        )
        for row in result["observation_telemetry"]
    } == {
        (
            candidate.name,
            candidate.host_cpu_budget,
            candidate.host_cpu_budget // candidate.total_concurrency,
            (
                candidate.host_cpu_budget
                // candidate.total_concurrency
                * candidate.total_concurrency
            ),
        )
        for candidate in config.candidates
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


def test_cpu_only_observation_explicitly_omits_gpu_lane_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _one_owner_plan()
    base = _config(
        fit_row_count=plan.physical_scopes[0].fit_row_count
    )
    candidate = replace(
        base.candidates[0],
        name="configured-cpu-only",
        accelerator_count=0,
        concurrency_per_device=1,
        neural_query_topology=_single_device_topology(),
    )
    scope = replace(
        base.representative_scopes[0],
        fits_per_observation=1,
    )
    workload = _tracked_workload(
        plan=plan,
        preflight_source=None,
        sessions=[],
        live_sessions={"current": 0, "maximum": 0},
    )
    _patch_measurement_telemetry(monkeypatch)

    _observation, detail, instances = (
        benchmark_module._run_observation(
            root=(tmp_path / "cpu-only-observation").resolve(),
            config=base,
            candidate=candidate,
            scope=scope,
            workload=workload,
            repetition_index=0,
            inventory=ResourceInventory(cpu_count=8, gpus=()),
            physical_owner_executor=(
                LocalThreadRoleNeutralPhysicalOwnerExecutor()
            ),
        )
    )

    assert detail["cpu_gpu_lane_interval_telemetry_required"] is False
    assert detail["cpu_gpu_lane_interval_telemetry_complete"] is True
    assert detail["cpu_gpu_lane_interval_observation"] is None
    assert detail["cpu_gpu_lane_overlap_analysis"] is None
    assert detail["cpu_gpu_lane_overlap_descriptive_only"] is False
    assert detail["cpu_gpu_lane_overlap_speedup_claimed"] is False
    assert len(instances) == 1
    assert len(instances[0].component_execution_intervals) == len(
        execution_support.EXPECTED_COMPONENT_FAMILIES
    )
    assert all(
        row["lane_kind"] == "cpu"
        and row["resource_ids"] == ["host_cpu"]
        for row in instances[0].component_execution_intervals
    )


def test_interrupted_benchmark_resumes_authenticated_complete_observations(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(fit_row_count=plan.physical_scopes[0].fit_row_count)
    _audit, _request, preflight_source = _seal(tmp_path)
    sessions: list = []
    live_sessions = {"current": 0, "maximum": 0}
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=sessions,
        live_sessions=live_sessions,
    )
    _patch_measurement_telemetry(monkeypatch)
    output_root = (tmp_path / "resumable-benchmark").resolve()
    original_run_observation = benchmark_module._run_observation
    first_run_calls = {"count": 0}

    def interrupt_third_observation(**kwargs):
        if first_run_calls["count"] == 2:
            original_run_observation(**kwargs)
            first_run_calls["count"] += 1
            raise KeyboardInterrupt("injected benchmark interruption")
        first_run_calls["count"] += 1
        return original_run_observation(**kwargs)

    monkeypatch.setattr(
        benchmark_module,
        "_run_observation",
        interrupt_third_observation,
    )
    with pytest.raises(KeyboardInterrupt, match="injected"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
        )
    assert first_run_calls["count"] == 3
    assert len(tuple((output_root / "checkpoints").iterdir())) == 2
    assert all(session.closed for session in sessions)
    assert live_sessions == {"current": 0, "maximum": 1}

    resume_events: list[tuple[str, str]] = []
    original_validate = (
        benchmark_module.validate_role_neutral_stage1_execution
    )

    def authenticated_validate(**kwargs):
        resume_events.append(("validate", str(kwargs["root"])))
        return original_validate(**kwargs)

    resumed_calls = {"count": 0}

    def count_new_observations(**kwargs):
        resume_events.append(
            ("execute", str(kwargs["candidate"].name))
        )
        resumed_calls["count"] += 1
        return original_run_observation(**kwargs)

    monkeypatch.setattr(
        benchmark_module,
        "validate_role_neutral_stage1_execution",
        authenticated_validate,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_run_observation",
        count_new_observations,
    )
    result = run_role_neutral_performance_benchmark(
        config=config,
        workloads={workload.scope_label: workload},
        output_root=output_root,
        inventory=_inventory(),
        resume=True,
    )

    expected_observation_count = (
        len(config.candidates)
        * len(config.representative_scopes)
        * (
            config.warmup_observations_per_candidate_scope
            + config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
        )
    )
    assert result["accepted"] is True
    assert resumed_calls["count"] == expected_observation_count - 2
    assert (
        len(tuple((output_root / "checkpoints").iterdir()))
        == expected_observation_count
    )
    assert len(
        tuple((output_root / "interrupted_observations").iterdir())
    ) == 2
    first_execution = next(
        index
        for index, event in enumerate(resume_events)
        if event[0] == "execute"
    )
    assert sum(
        event[0] == "validate"
        for event in resume_events[:first_execution]
    ) == 8
    assert all(session.closed for session in sessions)
    assert live_sessions == {"current": 0, "maximum": 1}
    assert [
        row["sequence_index"]
        for row in result["execution_schedule"]["entries"]
    ] == list(range(expected_observation_count))


def test_resume_rebinds_reordered_lane_intervals_to_reopened_fit_bytes(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(
        fit_row_count=plan.physical_scopes[0].fit_row_count
    )
    _audit, _request, preflight_source = _seal(tmp_path)
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=[],
        live_sessions={"current": 0, "maximum": 0},
    )
    _patch_measurement_telemetry(monkeypatch)
    output_root = (tmp_path / "lane-rebind-resume").resolve()
    paused = run_role_neutral_performance_benchmark(
        config=config,
        workloads={workload.scope_label: workload},
        output_root=output_root,
        inventory=_inventory(),
        stop_after_completed_observations=1,
    )
    assert paused["status"] == "paused"
    checkpoint_path = (
        output_root / "checkpoints" / "observation_000000.json"
    )
    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )
    interval_observation = checkpoint["detail"][
        "cpu_gpu_lane_interval_observation"
    ]
    intervals = interval_observation["intervals"]
    intervals[0], intervals[1] = intervals[1], intervals[0]
    interval_observation["content_sha256"] = (
        benchmark_module.identity_sha256(
            {
                key: value
                for key, value in interval_observation.items()
                if key != "content_sha256"
            }
        )
    )
    analysis = checkpoint["detail"]["cpu_gpu_lane_overlap_analysis"]
    analysis["source_observation_content_sha256"] = (
        interval_observation["content_sha256"]
    )
    analysis["content_sha256"] = benchmark_module.identity_sha256(
        {
            key: value
            for key, value in analysis.items()
            if key != "content_sha256"
        }
    )
    checkpoint["content_sha256"] = benchmark_module.identity_sha256(
        {
            key: value
            for key, value in checkpoint.items()
            if key != "content_sha256"
        }
    )
    checkpoint_path.chmod(0o600)
    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path.chmod(0o444)

    with pytest.raises(
        ValueError,
        match="differs from reopened fit intervals",
    ):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
            resume=True,
        )


def test_benchmark_resume_rejects_changed_immutable_request(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(fit_row_count=plan.physical_scopes[0].fit_row_count)
    _audit, _request, preflight_source = _seal(tmp_path)
    sessions: list = []
    live_sessions = {"current": 0, "maximum": 0}
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=sessions,
        live_sessions=live_sessions,
    )
    _patch_measurement_telemetry(monkeypatch)
    monkeypatch.setattr(
        benchmark_module,
        "_run_observation",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected before first observation")
        ),
    )
    output_root = (tmp_path / "changed-request-benchmark").resolve()
    with pytest.raises(RuntimeError, match="injected"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
        )

    with pytest.raises(FileExistsError, match="must be fresh"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
        )
    changed = replace(config, gpu_sample_interval_seconds=0.02)
    with pytest.raises(
        ValueError,
        match="identical immutable request",
    ):
        run_role_neutral_performance_benchmark(
            config=changed,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
            resume=True,
        )
    assert not tuple((output_root / "checkpoints").iterdir())


def test_persistent_session_is_closed_when_observation_fails(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    base_config = _config(
        fit_row_count=plan.physical_scopes[0].fit_row_count
    )
    persistent_first = next(
        candidate
        for candidate in base_config.candidates
        if candidate.name == "configured-single-persistent"
    )
    config = replace(
        base_config,
        candidates=(
            persistent_first,
            *tuple(
                candidate
                for candidate in base_config.candidates
                if candidate is not persistent_first
            ),
        ),
    )
    _audit, _request, preflight_source = _seal(tmp_path)
    sessions: list = []
    live_sessions = {"current": 0, "maximum": 0}
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=sessions,
        live_sessions=live_sessions,
    )
    _patch_measurement_telemetry(monkeypatch)
    monkeypatch.setattr(
        benchmark_module,
        "_run_observation",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected persistent observation failure")
        ),
    )

    with pytest.raises(RuntimeError, match="persistent observation"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=(tmp_path / "session-failure-benchmark").resolve(),
            inventory=_inventory(),
        )
    assert len(sessions) == 1
    assert sessions[0].closed is True
    assert live_sessions == {"current": 0, "maximum": 1}


def test_operational_observation_stop_pauses_and_resumes_same_request(
    tmp_path: Path,
    monkeypatch,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(fit_row_count=plan.physical_scopes[0].fit_row_count)
    _audit, _request, preflight_source = _seal(tmp_path)
    sessions: list = []
    live_sessions = {"current": 0, "maximum": 0}
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=sessions,
        live_sessions=live_sessions,
    )
    _patch_measurement_telemetry(monkeypatch)
    original_run_observation = benchmark_module._run_observation
    execution_calls: list[str] = []

    def track_observation(**kwargs):
        execution_calls.append(str(kwargs["candidate"].name))
        return original_run_observation(**kwargs)

    monkeypatch.setattr(
        benchmark_module,
        "_run_observation",
        track_observation,
    )
    output_root = (tmp_path / "paused-benchmark").resolve()
    paused = run_role_neutral_performance_benchmark(
        config=config,
        workloads={workload.scope_label: workload},
        output_root=output_root,
        inventory=_inventory(),
        stop_after_completed_observations=2,
    )
    request_bytes = (output_root / "benchmark_request.json").read_bytes()

    assert paused["schema_version"] == (
        benchmark_module.ROLE_NEUTRAL_BENCHMARK_PAUSED_RESULT_SCHEMA
    )
    assert paused["status"] == "paused"
    assert paused["completed_observation_count"] == 2
    assert paused["next_sequence_index"] == 2
    assert paused["terminal_benchmark_result_published"] is False
    assert paused["operational_stop_excluded_from_request_identity"] is True
    assert paused["content_sha256"] == benchmark_module.identity_sha256(
        {
            key: value
            for key, value in paused.items()
            if key != "content_sha256"
        }
    )
    assert execution_calls == [
        config.candidates[0].name,
        config.candidates[1].name,
    ]
    assert len(tuple((output_root / "checkpoints").iterdir())) == 2
    assert not (output_root / "benchmark_result.json").exists()
    assert all(session.closed for session in sessions)
    assert live_sessions == {"current": 0, "maximum": 1}

    execution_calls.clear()
    resumed_pause = run_role_neutral_performance_benchmark(
        config=config,
        workloads={workload.scope_label: workload},
        output_root=output_root,
        inventory=_inventory(),
        resume=True,
        stop_after_completed_observations=3,
    )
    assert resumed_pause["status"] == "paused"
    assert resumed_pause["completed_observation_count"] == 3
    assert execution_calls == [config.candidates[2].name]
    assert len(tuple((output_root / "checkpoints").iterdir())) == 3
    assert (output_root / "benchmark_request.json").read_bytes() == request_bytes
    assert not (output_root / "benchmark_result.json").exists()
    assert all(session.closed for session in sessions)
    assert live_sessions == {"current": 0, "maximum": 1}
    with pytest.raises(ValueError, match="precedes already sealed"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={workload.scope_label: workload},
            output_root=output_root,
            inventory=_inventory(),
            resume=True,
            stop_after_completed_observations=2,
        )


@pytest.mark.parametrize("invalid", (0, -1, True, 1.5))
def test_operational_observation_stop_must_be_positive_integer(
    tmp_path: Path,
    invalid,
) -> None:
    config = _config(fit_row_count=1)
    with pytest.raises(ValueError, match="positive integer"):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={},
            output_root=(tmp_path / f"invalid-{invalid}").resolve(),
            inventory=_inventory(),
            stop_after_completed_observations=invalid,
        )


def test_benchmark_cli_exposes_explicit_resume_flag(tmp_path: Path) -> None:
    from scripts.run_role_neutral_performance_benchmark import build_parser

    parsed = build_parser().parse_args(
        [
            "--benchmark-config",
            str(tmp_path / "benchmark.json"),
            "--workload-deployment",
            str(tmp_path / "workload.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--resume",
            "--stop-after-observations",
            "2",
            "--durable-publication-root",
            str(tmp_path / "durable"),
        ]
    )
    assert parsed.resume is True
    assert parsed.stop_after_observations == 2
    assert parsed.durable_publication_root == tmp_path / "durable"
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--benchmark-config",
                str(tmp_path / "benchmark.json"),
                "--workload-deployment",
                str(tmp_path / "workload.json"),
                "--output-root",
                str(tmp_path / "output"),
                "--stop-after-observations",
                "0",
            ]
        )


def test_checked_nsclc_matrix_is_typed_and_sizes_are_not_in_runner_source() -> None:
    path = (
        Path(__file__).parents[1]
        / "example_configs"
        / "portable_role_neutral_performance_benchmark_nsclc.deployment.json"
    )
    config = RoleNeutralBenchmarkConfig.from_json(path)
    assert config.schema_version == ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA
    assert len(config.representative_scopes) == 2
    assert len(config.candidates) == 8
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
    scientific_path = (
        Path(__file__).parents[1]
        / "example_configs"
        / "portable_all_evidence_scientific_nsclc.json"
    )
    scientific = json.loads(scientific_path.read_text(encoding="utf-8"))
    htr_science = scientific["architecture_profiles"][
        "hierarchical_transformer"
    ]["producer_configuration"]
    assert {
        value.htr_operational_controls.training_batch_size
        for value in config.candidates
    } == {htr_science["batch_size"]}
    import pyarrow.parquet as pq

    cohort_rows = pq.ParquetFile(
        Path(__file__).parents[1]
        / "synthetic_data"
        / "example_synthetic_datasets"
        / "one_confounder_one_effect_modifier_nsclc_with_structured"
        / "dataset.parquet"
    ).metadata.num_rows
    reusable_controls = [
        value.htr_operational_controls
        for value in config.candidates
        if value.htr_operational_controls.reuse_tokenizer_and_chunk_plans
    ]
    assert reusable_controls
    assert all(
        value.chunk_plan_cache_max_entries >= cohort_rows
        and value.tokenized_chunk_cache_max_entries
        >= cohort_rows * htr_science["max_chunks"]
        for value in reusable_controls
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


def test_benchmark_rejects_optimizer_batch_different_from_prepared_science(
    tmp_path: Path,
    portable_validators,
) -> None:
    plan = _one_owner_plan()
    config = _config(
        fit_row_count=plan.physical_scopes[0].fit_row_count,
    )
    _audit, _request, preflight_source = _seal(tmp_path)
    workload = _tracked_workload(
        plan=plan,
        preflight_source=preflight_source,
        sessions=[],
        live_sessions={"current": 0, "maximum": 0},
    )
    mismatched = replace(
        workload,
        scientific_htr_training_batch_size=5,
    )
    destination = (tmp_path / "mismatched-htr-batch").resolve()
    with pytest.raises(
        ValueError,
        match="authenticated prepared scientific profile",
    ):
        run_role_neutral_performance_benchmark(
            config=config,
            workloads={mismatched.scope_label: mismatched},
            output_root=destination,
            inventory=_inventory(),
        )
    assert not destination.exists()


def test_process_isolated_fit_counters_are_closed_and_attributed_once() -> None:
    manifest = {
        "owner_execution_telemetry": {
            "process_isolated_physical_owners": True,
            "parent_process_counters_included_in_child_counters": False,
            "physical_owners": [
                {
                    "resource": "cuda:4",
                    "telemetry": {
                        "schema_version": (
                            "production_role_neutral_process_owner_telemetry_v1"
                        ),
                        "resource": "cuda:4",
                        "reserved_resources": ["cuda:4"],
                        "wall_seconds": 1.25,
                        "cpu_seconds": 0.75,
                        "peak_gpu_allocated_bytes": 505,
                        "peak_gpu_reserved_bytes": 606,
                        "peak_gpu_allocated_bytes_by_device": {
                            "cuda:4": 505,
                        },
                        "peak_gpu_reserved_bytes_by_device": {
                            "cuda:4": 606,
                        },
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
        {"cuda:4": 606},
    )
    manifest["owner_execution_telemetry"]["physical_owners"][0][
        "telemetry"
    ]["process_io_deltas"].pop("rchar")
    with pytest.raises(ValueError, match="closed child I/O"):
        benchmark_module._child_process_io_from_manifest(manifest)


def test_process_isolated_gpu_peaks_are_closed_over_every_reserved_device() -> None:
    telemetry = {
        "schema_version": "production_role_neutral_process_owner_telemetry_v1",
        "resource": "cuda:4",
        "reserved_resources": ["cuda:4", "cuda:9"],
        "wall_seconds": 1.25,
        "cpu_seconds": 0.75,
        "peak_gpu_allocated_bytes": 505,
        "peak_gpu_reserved_bytes": 606,
        "peak_gpu_allocated_bytes_by_device": {
            "cuda:4": 505,
            "cuda:9": 707,
        },
        "peak_gpu_reserved_bytes_by_device": {
            "cuda:4": 606,
            "cuda:9": 808,
        },
        "process_io_deltas": {
            "rchar": 101,
            "wchar": 202,
            "read_bytes": 303,
            "write_bytes": 404,
        },
    }
    manifest = {
        "owner_execution_telemetry": {
            "process_isolated_physical_owners": True,
            "parent_process_counters_included_in_child_counters": False,
            "physical_owners": [
                {
                    "resource": "cuda:4",
                    "telemetry": telemetry,
                }
            ],
        }
    }
    assert benchmark_module._child_process_io_from_manifest(manifest)[-1] == {
        "cuda:4": 606,
        "cuda:9": 808,
    }

    telemetry["peak_gpu_reserved_bytes_by_device"].pop("cuda:9")
    with pytest.raises(ValueError, match="closed per-device GPU peaks"):
        benchmark_module._child_process_io_from_manifest(manifest)


def test_spanning_child_peaks_are_aggregated_on_each_reserved_device() -> None:
    instances = (
        SimpleNamespace(
            complete_record=SimpleNamespace(gpu_peak_memory_bytes={}),
            child_peak_gpu_memory_bytes_by_device={
                "cuda:4": 200,
                "cuda:9": 300,
            },
        ),
        SimpleNamespace(
            complete_record=SimpleNamespace(gpu_peak_memory_bytes={}),
            child_peak_gpu_memory_bytes_by_device={
                "cuda:4": 400,
                "cuda:9": 100,
            },
        ),
    )
    samples = (
        {
            "device": "cuda:4",
            "memory_used_bytes": 100,
            "memory_total_bytes": 1_000,
        },
        {
            "device": "cuda:9",
            "memory_used_bytes": 100,
            "memory_total_bytes": 1_000,
        },
    )
    peak, headroom, complete, by_device = (
        benchmark_module._candidate_memory_observation(
            devices=("cuda:4", "cuda:9"),
            inventory=_inventory(),
            samples=samples,
            instances=instances,
            concurrency_per_device=2,
        )
    )
    assert complete is True
    assert peak == pytest.approx(0.7)
    assert headroom == 300
    assert by_device == {
        "cuda:4": 700,
        "cuda:9": 500,
    }


def test_cpu_and_local_gpu_peak_aggregation_remain_device_neutral() -> None:
    assert benchmark_module._candidate_memory_observation(
        devices=("cpu",),
        inventory=ResourceInventory(cpu_count=2, gpus=()),
        samples=(),
        instances=(),
        concurrency_per_device=1,
    ) == (None, None, True, {})

    local_instance = SimpleNamespace(
        complete_record=SimpleNamespace(
            gpu_peak_memory_bytes={"cuda:4": 250},
        ),
        child_peak_gpu_memory_bytes_by_device=None,
    )
    peak, headroom, complete, by_device = (
        benchmark_module._candidate_memory_observation(
            devices=("cuda:4",),
            inventory=_inventory(),
            samples=(
                {
                    "device": "cuda:4",
                    "memory_used_bytes": 100,
                    "memory_total_bytes": 1_000,
                },
            ),
            instances=(local_instance,),
            concurrency_per_device=1,
        )
    )
    assert (peak, headroom, complete, by_device) == (
        pytest.approx(0.25),
        750,
        True,
        {"cuda:4": 250},
    )


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
            neural_query_topology=_single_device_topology(),
            htr_operational_controls=RoleNeutralHTROperationalControls(
                training_batch_size=4,
                sentence_encoder_batch_size=8,
                data_loader_workers=0,
                reuse_tokenizer_and_chunk_plans=False,
                chunk_plan_cache_max_entries=0,
                tokenized_chunk_cache_max_entries=0,
            ),
        )
    with pytest.raises(ValueError, match="traversal-safe"):
        RoleNeutralBenchmarkScope(
            label=unsafe,
            fit_row_count=1,
            fits_per_observation=1,
        )
