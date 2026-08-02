from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import pytest

import oci.inference.production_all_evidence_workflow as workflow_module
from oci.inference.neural_query_operational_controls import (
    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA,
    RoleNeutralNeuralQueryOperationalControls,
)
from oci.inference.portable_workflow_spec import (
    DeploymentProfile,
    Stage1ExecutionProfile,
    Stage1OwnerCapacityPolicy,
    Stage1PreflightExecutionPolicy,
)
from oci.inference.portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
    ResourcePlan,
    resolve_stage1_owner_capacity,
)
from oci.inference.production_role_neutral_persistent_executor import (
    PersistentRoleNeutralExecutionSession,
)
from oci.inference.production_role_neutral_process_executor import (
    _task_execution_resources,
)
from oci.inference.production_stage1_role_neutral_coordinator import (
    ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION,
)
from oci.inference.production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    NeuralQueryExecutionTopology,
    ROLE_NEUTRAL_COORDINATION_DIRECTORY,
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    RoleNeutralOperationalComponentReport,
    RoleNeutralPhysicalOwnerResult,
    RoleNeutralPhysicalOwnerTask,
    RoleNeutralProducerFactories,
    RoleNeutralStage1ExecutionPolicy,
    _execute_one_owner,
    _prior_authenticated_component_receipt,
    _write_new_json,
    execute_and_publish_role_neutral_stage1,
    validate_role_neutral_stage1_execution,
)
from oci.inference.role_neutral_all_ten_binding import (
    EXPECTED_COMPONENT_FAMILIES,
    authenticated_role_neutral_component_tree_sha256,
    validate_authenticated_role_neutral_component_receipt,
)
from oci.inference.stage1_execution_topology_policy import (
    ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
)
from oci.inference.stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)
from tests.stage1_test_support import stage1_execution_profile
from tests.test_portable_workflow_contracts import (
    _forest_operational,
    _resource_safety,
)
from tests.test_production_stage1_role_neutral_execution import (
    _ProducerRecorder,
    _RecordingExecutor,
    _plan,
    _resource_plan,
    _sha,
)


def _deployment(
    tmp_path: Path,
    *,
    execution,
    devices: tuple[str, ...],
    cpu_budget: int,
) -> DeploymentProfile:
    profile = DeploymentProfile(
        dataset_path=tmp_path / "cohort.parquet",
        durable_artifact_root=tmp_path / "artifacts",
        scratch_root=tmp_path / "scratch",
        embedding_model_locator=tmp_path / "embedding-model",
        htr_model_locator=tmp_path / "htr-model",
        stage1_profile_locator=tmp_path / "stage1.json",
        query_profile_locator=tmp_path / "query.json",
        embedding_batch_size=4,
        resource_performance_safety=_resource_safety(),
        forest_operational=_forest_operational(cpu_budget),
        stage1_execution=execution,
        cluster_preflight_parquet_compression="zstd",
        devices=devices,
        cpu_budget=cpu_budget,
    )
    return DeploymentProfile.from_mapping(asdict(profile))


def test_deployment_compilation_bounds_generic_owner_capacity(
    tmp_path: Path,
) -> None:
    cpu = _deployment(
        tmp_path / "cpu",
        execution=stage1_execution_profile(
            resource_kind="cpu",
            device_count=1,
            scope_workers_per_device=4,
            max_parallel_owners=2,
        ),
        devices=("cpu",),
        cpu_budget=4,
    )
    one_device = _deployment(
        tmp_path / "one-device",
        execution=stage1_execution_profile(
            resource_kind="accelerator",
            device_count=1,
            scope_workers_per_device=2,
            max_parallel_owners=2,
        ),
        devices=("cuda:7",),
        cpu_budget=4,
    )
    multi_device = _deployment(
        tmp_path / "multi-device",
        execution=stage1_execution_profile(
            resource_kind="accelerator",
            device_count=3,
            scope_workers_per_device=2,
            max_parallel_owners=3,
        ),
        devices=("cuda:1", "cuda:4", "cuda:9"),
        cpu_budget=6,
    )
    spanning = _deployment(
        tmp_path / "spanning",
        execution=stage1_execution_profile(
            resource_kind="accelerator",
            device_count=3,
            scope_workers_per_device=2,
            max_parallel_owners=2,
            topology_mode=(
                ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES
            ),
            neural_query_inner_fold_parallelism=3,
            neural_query_fold_parallel_backend="processes",
            neural_query_bank_parallelism=3,
        ),
        devices=("cuda:2", "cuda:5", "cuda:8"),
        cpu_budget=6,
    )

    assert cpu.stage1_execution.max_parallel_owners == 2
    assert one_device.stage1_execution.max_parallel_owners == 2
    assert multi_device.stage1_execution.max_parallel_owners == 3
    assert (
        multi_device.stage1_execution.scope_workers_per_device
        * multi_device.stage1_execution.device_count
        == 6
    )
    assert spanning.stage1_execution.max_parallel_owners == 2

    with pytest.raises(ValueError, match="topology capacity"):
        stage1_execution_profile(
            resource_kind="accelerator",
            device_count=3,
            scope_workers_per_device=1,
            max_parallel_owners=4,
        )

    cpu_overcommit = stage1_execution_profile(
        resource_kind="cpu",
        device_count=1,
        scope_workers_per_device=3,
        max_parallel_owners=3,
    )
    with pytest.raises(ValueError, match="global host CPU budget"):
        _deployment(
            tmp_path / "cpu-overcommit",
            execution=cpu_overcommit,
            devices=("cpu",),
            cpu_budget=2,
        )

    one_device_controls = one_device.stage1_execution.htr_operational_controls
    with pytest.raises(ValueError, match="per-owner lease"):
        replace(
            one_device.stage1_execution,
            htr_operational_controls=replace(
                one_device_controls,
                fold_parallelism=2,
                fold_parallel_backend="processes",
            ),
        )


def test_new_profiles_default_to_autodetect_but_legacy_profiles_stay_fixed(
) -> None:
    current = stage1_execution_profile(
        resource_kind="cpu",
        device_count=1,
        scope_workers_per_device=1,
    )
    assert current.owner_capacity_policy.mode == "resource_autodetect"

    current_mapping = asdict(current)
    current_mapping.pop("owner_capacity_policy")
    with pytest.raises(
        ValueError,
        match="must explicitly configure owner_capacity_policy",
    ):
        Stage1ExecutionProfile.from_mapping(current_mapping)

    legacy_mapping = dict(current_mapping)
    legacy_mapping["schema_version"] = (
        "portable_stage1_execution_profile_v8"
    )
    migrated = Stage1ExecutionProfile.from_mapping(legacy_mapping)
    assert migrated.owner_capacity_policy.mode == "fixed"


def test_runtime_owner_capacity_autodetects_vram_host_and_cpu_caps() -> None:
    gib = 1024**3
    configured = replace(
        stage1_execution_profile(
            resource_kind="accelerator",
            device_count=8,
            scope_workers_per_device=4,
            max_parallel_owners=32,
        ),
        owner_capacity_policy=Stage1OwnerCapacityPolicy(
            mode="resource_autodetect",
            estimated_device_memory_bytes_per_owner=8 * gib,
            device_memory_reserve_bytes=6 * gib,
            estimated_host_memory_bytes_per_owner=8 * gib,
            host_memory_budget_fraction=0.75,
            minimum_cpu_threads_per_owner=1,
        ),
        preflight_execution_policy=Stage1PreflightExecutionPolicy(
            max_parallel_owners=8,
            memory_budget_bytes=64 * gib,
            estimated_owner_peak_bytes=8 * gib,
            input_io_lane_cap=8,
            publication_io_lane_cap=8,
            authentication_io_lane_cap=8,
        ),
    )
    resources = ResourceInventory(
        cpu_count=64,
        gpus=tuple(
            GPUResource(
                device=f"cuda:{index}",
                uuid=f"gpu-{index}",
                total_memory_bytes=96 * gib,
                free_memory_bytes=96 * gib,
                utilization_percent=0.0,
            )
            for index in range(8)
        ),
    )
    plan = ResourcePlan(
        devices=tuple(f"cuda:{index}" for index in range(8)),
        cpu_budget=64,
        inventory=resources,
        policy=("auto",),
        resource_performance_safety=_resource_safety(
            maximum_allocation_fraction=0.85,
        ),
    )

    effective, attestation = resolve_stage1_owner_capacity(
        profile=configured,
        resource_plan=plan,
        host_available_memory_bytes=512 * gib,
    )

    assert effective.scope_workers_per_device == 4
    assert effective.max_parallel_owners == 32
    assert attestation["mode"] == "resource_autodetect"
    assert attestation["minimum_uniform_device_lane_cap"] == 11
    assert attestation["effective_scope_workers_per_device"] == 4
    assert attestation["effective_max_parallel_owners"] == 32
    assert len(attestation["per_device_capacity"]) == 8

    host_limited, host_attestation = resolve_stage1_owner_capacity(
        profile=configured,
        resource_plan=plan,
        host_available_memory_bytes=128 * gib,
    )
    assert host_limited.scope_workers_per_device == 4
    assert host_limited.max_parallel_owners == 12
    assert host_attestation["host_owner_lane_cap"] == 12
    assert (
        host_limited.preflight_execution_policy.max_parallel_owners
        == 8
    )


def test_runtime_owner_capacity_allows_global_cap_below_device_count() -> None:
    gib = 1024**3
    configured = stage1_execution_profile(
        resource_kind="accelerator",
        device_count=4,
        scope_workers_per_device=4,
        max_parallel_owners=2,
    )
    inventory = ResourceInventory(
        cpu_count=16,
        gpus=tuple(
            GPUResource(
                device=f"cuda:{index}",
                uuid=f"gpu-{index}",
                total_memory_bytes=96 * gib,
                free_memory_bytes=96 * gib,
                utilization_percent=0.0,
            )
            for index in range(4)
        ),
    )
    plan = ResourcePlan(
        devices=tuple(f"cuda:{index}" for index in range(4)),
        cpu_budget=16,
        inventory=inventory,
        policy=("auto",),
        resource_performance_safety=_resource_safety(
            maximum_allocation_fraction=0.85,
        ),
    )

    effective, attestation = resolve_stage1_owner_capacity(
        profile=configured,
        resource_plan=plan,
        host_available_memory_bytes=512 * gib,
    )

    assert effective.scope_workers_per_device == 4
    assert effective.max_parallel_owners == 2
    assert attestation["topology_owner_lane_cap"] == 16
    assert attestation["effective_max_parallel_owners"] == 2


def test_runtime_owner_capacity_uses_smallest_selected_gpu_and_fixed_fallback() -> None:
    gib = 1024**3
    configured = replace(
        stage1_execution_profile(
            resource_kind="accelerator",
            device_count=2,
            scope_workers_per_device=4,
            max_parallel_owners=8,
        ),
        owner_capacity_policy=Stage1OwnerCapacityPolicy(
            mode="resource_autodetect",
            estimated_device_memory_bytes_per_owner=8 * gib,
            device_memory_reserve_bytes=6 * gib,
            estimated_host_memory_bytes_per_owner=4 * gib,
            host_memory_budget_fraction=0.75,
            minimum_cpu_threads_per_owner=1,
        ),
        preflight_execution_policy=Stage1PreflightExecutionPolicy(
            max_parallel_owners=4,
            memory_budget_bytes=32 * gib,
            estimated_owner_peak_bytes=8 * gib,
            input_io_lane_cap=4,
            publication_io_lane_cap=4,
            authentication_io_lane_cap=4,
        ),
    )
    inventory = ResourceInventory(
        cpu_count=32,
        gpus=(
            GPUResource(
                device="cuda:0",
                uuid="large",
                total_memory_bytes=96 * gib,
                free_memory_bytes=96 * gib,
                utilization_percent=0.0,
            ),
            GPUResource(
                device="cuda:1",
                uuid="small",
                total_memory_bytes=24 * gib,
                free_memory_bytes=20 * gib,
                utilization_percent=0.0,
            ),
        ),
    )
    plan = ResourcePlan(
        devices=("cuda:0", "cuda:1"),
        cpu_budget=16,
        inventory=inventory,
        policy=("cuda:0", "cuda:1"),
        resource_performance_safety=_resource_safety(
            maximum_allocation_fraction=0.85,
        ),
    )

    effective, attestation = resolve_stage1_owner_capacity(
        profile=configured,
        resource_plan=plan,
        host_available_memory_bytes=256 * gib,
    )
    assert effective.scope_workers_per_device == 1
    assert effective.max_parallel_owners == 2
    assert attestation["minimum_uniform_device_lane_cap"] == 1

    fixed = replace(
        configured,
        owner_capacity_policy=Stage1OwnerCapacityPolicy(
            mode="fixed",
            estimated_device_memory_bytes_per_owner=1,
            device_memory_reserve_bytes=0,
            estimated_host_memory_bytes_per_owner=1,
            host_memory_budget_fraction=1.0,
            minimum_cpu_threads_per_owner=1,
        ),
    )
    fixed_effective, fixed_attestation = (
        resolve_stage1_owner_capacity(
            profile=fixed,
            resource_plan=plan,
            host_available_memory_bytes=1,
        )
    )
    assert fixed_effective == fixed
    assert fixed_attestation["mode"] == "fixed"


def test_deployment_compiles_independent_preflight_resource_caps(
    tmp_path: Path,
) -> None:
    execution = stage1_execution_profile(
        resource_kind="accelerator",
        device_count=4,
        scope_workers_per_device=1,
        max_parallel_owners=4,
    )
    policy = Stage1PreflightExecutionPolicy(
        max_parallel_owners=4,
        memory_budget_bytes=3_000,
        estimated_owner_peak_bytes=1_000,
        input_io_lane_cap=3,
        publication_io_lane_cap=2,
        authentication_io_lane_cap=4,
    )
    deployment = _deployment(
        tmp_path / "bounded",
        execution=replace(
            execution,
            preflight_execution_policy=policy,
        ),
        devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        cpu_budget=8,
    )
    deployment = DeploymentProfile.from_mapping(
        asdict(
            replace(
                deployment,
                resource_performance_safety=replace(
                    deployment.resource_performance_safety,
                    maximum_ordinary_read_amplification=4.0,
                ),
            )
        )
    )

    attestation = (
        workflow_module._stage1_preflight_execution_attestation(
            deployment
        )
    )
    assert attestation[
        "effective_preflight_owner_lanes_before_scope_cap"
    ] == 2
    assert attestation["derived_caps"] == {
        "cpu_budget": 8,
        "stage1_owner_cap": 4,
        "preflight_owner_cap": 4,
        "memory_lane_cap": 3,
        "input_io_lane_cap": 3,
        "publication_io_lane_cap": 2,
        "authentication_io_lane_cap": 4,
        "ordinary_read_amplification_lane_cap": 4,
    }
    assert attestation["resource_assignment_in_scientific_identity"] is False
    assert attestation["completion_order_in_scientific_identity"] is False

    # Profiles written before the preflight policy existed remain valid and
    # deliberately reopen with one conservative lane.
    legacy_mapping = asdict(execution)
    legacy_mapping.pop("preflight_execution_policy")
    legacy = type(execution).from_mapping(legacy_mapping)
    assert legacy.preflight_execution_policy.max_parallel_owners == 1
    assert legacy.preflight_execution_policy.memory_lane_cap == 1

    with pytest.raises(ValueError, match="estimated_owner_peak_bytes"):
        Stage1PreflightExecutionPolicy(
            max_parallel_owners=1,
            memory_budget_bytes=999,
            estimated_owner_peak_bytes=1_000,
            input_io_lane_cap=1,
            publication_io_lane_cap=1,
            authentication_io_lane_cap=1,
        )
    with pytest.raises(ValueError, match="preflight max_parallel_owners"):
        replace(
            execution,
            preflight_execution_policy=replace(
                policy,
                max_parallel_owners=5,
            ),
        )


class _OperationalProducerRecorder(_ProducerRecorder):
    def factory(self, expected_component: str):
        base_factory = super().factory(expected_component)

        def bind(invocation):
            bound = base_factory(invocation)
            if expected_component not in {
                "matched_pair",
                "neural_query",
            }:
                return bound

            def execute():
                bound.execute()
                body = {
                    "schema_version": (
                        "test_parallel_owner_operational_attestation_v1"
                    ),
                    "component": expected_component,
                    "physical_owner_scope_id": (
                        invocation.physical_owner.scope_id
                    ),
                }
                return RoleNeutralOperationalComponentReport(
                    component=expected_component,
                    attestation={
                        **body,
                        "content_sha256": _sha(body),
                    },
                )

            return replace(bound, execute=execute)

        return bind


def _wait_for(path: Path, *, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for disjoint owner marker {path}"
            )
        time.sleep(0.01)


def _attempt_factories(
    *,
    marker_root: Path,
    fail_owner: str | None,
) -> RoleNeutralProducerFactories:
    base = _OperationalProducerRecorder().factories()

    def wrap(component: str, factory):
        def bind(invocation):
            bound = factory(invocation)

            def execute():
                owner = invocation.physical_owner.scope_id
                owner_markers = marker_root / owner
                owner_markers.mkdir(parents=True, exist_ok=True)
                (owner_markers / f"execute-{component}").write_text(
                    "started\n",
                    encoding="utf-8",
                )
                if owner == fail_owner and component == "neural_query":
                    invocation.output_root.mkdir(
                        parents=True,
                        exist_ok=False,
                    )
                    (invocation.output_root / "partial.bin").write_bytes(
                        b"interrupted"
                    )
                    raise RuntimeError(
                        "intentional component interruption"
                    )
                result = bound.execute()
                (
                    owner_markers / f"completed-{component}"
                ).write_text("complete\n", encoding="utf-8")
                return result

            return replace(bound, execute=execute)

        return bind

    mapping = base.as_mapping()
    return RoleNeutralProducerFactories(
        **{
            component: wrap(component, mapping[component])
            for component in EXPECTED_COMPONENT_FAMILIES
        }
    )


class _InProcessConnection:
    def __init__(
        self,
        *,
        slot_index: int,
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
    ) -> None:
        self.slot_index = int(slot_index)
        self.worker = worker
        self.task: RoleNeutralPhysicalOwnerTask | None = None

    def send(self, message: Mapping[str, Any]) -> None:
        if (
            not isinstance(message, Mapping)
            or message.get("command") != "execute"
            or not isinstance(
                message.get("task"),
                RoleNeutralPhysicalOwnerTask,
            )
        ):
            raise AssertionError(
                "in-process persistent lane received another command"
            )
        self.task = message["task"]

    def recv(self) -> Mapping[str, Any]:
        task = self.task
        if task is None:
            raise AssertionError(
                "in-process persistent lane received before send"
            )
        self.task = None
        try:
            result = self.worker(task)
        except BaseException as exc:
            return {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": "in-process scheduler fixture",
            }
        return {
            "status": "completed",
            "result": result,
            "telemetry": {
                "schema_version": (
                    "test_in_process_persistent_owner_telemetry_v1"
                ),
                "slot_index": self.slot_index,
                "reserved_resources": list(
                    _task_execution_resources(task)
                ),
                "worker_report": result.execution_telemetry,
            },
        }


class _InProcessPersistentSession(
    PersistentRoleNeutralExecutionSession
):
    def __init__(
        self,
        *,
        resources: tuple[str, ...],
        max_workers: int,
        cpu_budget: int,
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
    ) -> None:
        self._condition = threading.Condition()
        self._closed = False
        self._broken: BaseException | None = None
        self._active_calls = 0
        self.max_parallel_owners = int(max_workers)
        self.host_cpu_budget = int(cpu_budget)
        self._slots = [
            SimpleNamespace(
                index=index,
                resource=resource,
                connection=_InProcessConnection(
                    slot_index=index,
                    worker=worker,
                ),
                busy=False,
            )
            for index, resource in enumerate(resources)
        ]

    def close(self) -> None:
        with self._condition:
            while self._active_calls:
                self._condition.wait()
            self._closed = True
            self._condition.notify_all()

    def interrupt(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _InProcessPersistentExecutor:
    process_isolated_physical_owners = True

    def __init__(
        self,
        *,
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
    ) -> None:
        self.worker = worker
        self.resource_leases: tuple[tuple[str, ...], ...] = ()

    def execute(self, **_kwargs):
        raise AssertionError(
            "coordinator bypassed the persistent execution session"
        )

    def open_session(
        self,
        *,
        resources,
        resource_leases,
        max_workers,
        cpu_budget,
        marker_root,
    ) -> _InProcessPersistentSession:
        del marker_root
        self.resource_leases = tuple(
            tuple(lease) for lease in resource_leases
        )
        return _InProcessPersistentSession(
            resources=tuple(resources),
            max_workers=int(max_workers),
            cpu_budget=int(cpu_budget),
            worker=self.worker,
        )


def _controlled_policy() -> RoleNeutralStage1ExecutionPolicy:
    devices = ("cuda:0", "cuda:1")
    return RoleNeutralStage1ExecutionPolicy(
        resource_plan=_resource_plan(
            devices=devices,
            cpu_budget=4,
        ),
        max_parallel_owners=2,
        neural_query_execution_topologies={
            device: NeuralQueryExecutionTopology.single(device)
            for device in devices
        },
        htr_operational_controls=RoleNeutralHTROperationalControls(
            training_batch_size=4,
            sentence_encoder_batch_size=8,
            data_loader_workers=0,
            fold_parallelism=1,
            fold_parallel_backend="threads",
            fold_slots_per_device=1,
            reuse_tokenizer_and_chunk_plans=True,
            chunk_plan_cache_max_entries=100,
            tokenized_chunk_cache_max_entries=1000,
        ),
        neural_query_operational_controls=(
            RoleNeutralNeuralQueryOperationalControls(
                inner_fold_parallelism=1,
                fold_parallel_backend="threads",
                fold_slots_per_device=1,
                bank_parallelism=1,
                worker_cpu_threads=1,
                schema_version=(
                    ROLE_NEUTRAL_NEURAL_QUERY_OPERATIONAL_CONTROLS_SCHEMA
                ),
            )
        ),
    )


def test_disjoint_owner_lanes_merge_canonically_and_resume_components(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=(0, 1))
    owner_zero, owner_one = plan.physical_execution_order[:2]
    component_store = (
        tmp_path / "stage1-component-store" / "components"
    ).resolve()
    first_markers = (tmp_path / "first-markers").resolve()
    interrupted_root = (
        tmp_path / "interrupted-execution"
    ).resolve()

    def first_worker(
        task: RoleNeutralPhysicalOwnerTask,
    ) -> RoleNeutralPhysicalOwnerResult:
        owner = task.physical_owner.scope_id
        owner_root = first_markers / owner
        owner_root.mkdir(parents=True, exist_ok=True)
        (owner_root / "owner-started").write_text(
            "started\n",
            encoding="utf-8",
        )
        if owner == owner_zero:
            _wait_for(
                first_markers / owner_one / "owner-completed"
            )
        result = _execute_one_owner(
            task=task,
            factories=_attempt_factories(
                marker_root=first_markers,
                fail_owner=owner_zero,
            ).as_mapping(),
        )
        (owner_root / "owner-completed").write_text(
            "complete\n",
            encoding="utf-8",
        )
        return result

    first_executor = _InProcessPersistentExecutor(
        worker=first_worker,
    )
    with pytest.raises(RuntimeError):
        execute_and_publish_role_neutral_stage1(
            root=interrupted_root,
            plan=plan,
            producer_factories=(
                _OperationalProducerRecorder().factories()
            ),
            policy=_controlled_policy(),
            executor=first_executor,
        )

    assert set(first_executor.resource_leases) == {
        ("cuda:0",),
        ("cuda:1",),
    }
    assert (
        first_markers / owner_zero / "owner-started"
    ).is_file()
    assert (
        first_markers / owner_one / "owner-completed"
    ).is_file()
    interrupted_owner = (
        interrupted_root / "components" / owner_zero
    )
    component_order = tuple(EXPECTED_COMPONENT_FAMILIES)
    for component in component_order[:-1]:
        assert (
            interrupted_owner
            / component
            / ROLE_NEUTRAL_EXECUTION_MANIFEST
        ).is_file()
    assert not (interrupted_owner / "neural_query").exists()
    assert tuple(
        interrupted_owner.glob(".neural_query.attempt-*")
    )

    second_markers = (tmp_path / "second-markers").resolve()

    def second_worker(
        task: RoleNeutralPhysicalOwnerTask,
    ) -> RoleNeutralPhysicalOwnerResult:
        owner = task.physical_owner.scope_id
        owner_root = second_markers / owner
        owner_root.mkdir(parents=True, exist_ok=True)
        (owner_root / "owner-started").write_text(
            "started\n",
            encoding="utf-8",
        )
        if owner == owner_zero:
            _wait_for(
                second_markers / owner_one / "owner-completed"
            )
        result = _execute_one_owner(
            task=task,
            factories=_attempt_factories(
                marker_root=second_markers,
                fail_owner=None,
            ).as_mapping(),
        )
        (owner_root / "owner-completed").write_text(
            "complete\n",
            encoding="utf-8",
        )
        return result

    resumed_executor = _InProcessPersistentExecutor(
        worker=second_worker,
    )
    resumed_root = interrupted_root
    manifest = execute_and_publish_role_neutral_stage1(
        root=resumed_root,
        plan=plan,
        producer_factories=(
            _OperationalProducerRecorder().factories()
        ),
        policy=_controlled_policy(),
        executor=resumed_executor,
        resume=True,
        component_store_root=component_store,
    )

    assert (
        validate_role_neutral_stage1_execution(
            root=resumed_root,
            plan=plan,
        )
        == manifest
    )
    assert sorted(
        path.name
        for path in (second_markers / owner_zero).glob(
            "execute-*"
        )
    ) == ["execute-neural_query"]
    assert not tuple(
        (second_markers / owner_one).glob("execute-*")
    )
    owner_store = component_store / owner_zero
    assert not tuple(owner_store.glob(".*.attempt-*"))
    assert not tuple(
        interrupted_owner.glob(".*.attempt-*")
    )
    assert tuple(
            (
                tmp_path
                / "interrupted_role_neutral_materializations"
                / owner_zero
            ).glob("neural_query.*")
        )

    attestation = json.loads(
        (
            resumed_root / "execution_attestation.json"
        ).read_text(encoding="utf-8")
    )
    completion_order = attestation["completed_owner_order"]
    assert completion_order.index(owner_one) < completion_order.index(
        owner_zero
    )
    assert completion_order != list(plan.physical_execution_order)
    assert (
        attestation["effective_owner_concurrency_policy"]
        == "configured_disjoint_owner_lease_capacity_v2"
    )
    assert attestation["effective_max_parallel_owners"] == 2

    telemetry_by_owner = {
        row["physical_owner_scope_id"]: row
        for row in attestation[
            "owner_execution_telemetry"
        ]["physical_owners"]
    }
    for owner, resource in (
        (owner_zero, "cuda:0"),
        (owner_one, "cuda:1"),
    ):
        telemetry = telemetry_by_owner[owner]["telemetry"]
        assert telemetry["reserved_resources"] == [resource]
        gpu_intervals = [
            row
            for row in telemetry["worker_report"][
                "component_execution_intervals"
            ]
            if row["lane_kind"] == "gpu"
        ]
        assert gpu_intervals
        assert all(
            row["resource_ids"] == [resource]
            for row in gpu_intervals
        )

    locator_attestation = json.loads(
        (
            resumed_root
            / ROLE_NEUTRAL_COORDINATION_DIRECTORY
            / ROLE_NEUTRAL_COMPONENT_LOCATOR_ATTESTATION
        ).read_text(encoding="utf-8")
    )
    assert locator_attestation[
        "physical_owner_scope_order"
    ] == [
        owner.scope_id for owner in plan.physical_scopes
    ]
    assert all(
        (
            resumed_root
            / "components"
            / owner.scope_id
            / component
            / ROLE_NEUTRAL_EXECUTION_MANIFEST
        ).is_file()
        for owner in plan.physical_scopes
        for component in EXPECTED_COMPONENT_FAMILIES
    )


def test_cross_request_component_import_reuses_only_currently_authenticated_work(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    policy = RoleNeutralStage1ExecutionPolicy(
        resource_plan=_resource_plan(
            devices=("cpu",),
            cpu_budget=4,
        ),
        max_parallel_owners=1,
    )
    legacy_execution = (tmp_path / "legacy-execution").resolve()
    execute_and_publish_role_neutral_stage1(
        root=legacy_execution,
        plan=plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=policy,
        executor=_RecordingExecutor(),
    )
    legacy_components = legacy_execution / "components"
    first_owner = plan.physical_execution_order[0]
    preserved_htr_terminal = (
        legacy_components
        / first_owner
        / "htr"
        / ROLE_NEUTRAL_EXECUTION_MANIFEST
    ).read_bytes()

    corrected = _ProducerRecorder()
    corrected_factories = corrected.factories()

    def corrected_htr_factory(invocation):
        bound = corrected_factories.htr(invocation)

        def authenticate():
            if (
                legacy_components in invocation.output_root.parents
                and invocation.physical_owner.scope_id == first_owner
            ):
                raise ValueError(
                    "legacy HTR is scientifically incompatible"
                )
            return bound.authenticate()

        return replace(bound, authenticate=authenticate)

    target_store = (
        tmp_path / "corrected-component-store" / "components"
    ).resolve()
    corrected_execution = (tmp_path / "corrected-execution").resolve()

    class _LaneImportExecutor(_RecordingExecutor):
        def execute(self, *, tasks, worker, max_workers, cpu_budget):
            assert not tuple(
                target_store.rglob(ROLE_NEUTRAL_EXECUTION_MANIFEST)
            )
            return super().execute(
                tasks=tasks,
                worker=worker,
                max_workers=max_workers,
                cpu_budget=cpu_budget,
            )

    manifest = execute_and_publish_role_neutral_stage1(
        root=corrected_execution,
        plan=plan,
        producer_factories=RoleNeutralProducerFactories(
            bow=corrected_factories.bow,
            htr=corrected_htr_factory,
            matched_pair=corrected_factories.matched_pair,
            embeddings=corrected_factories.embeddings,
            tfidf=corrected_factories.tfidf,
            neural_query=corrected_factories.neural_query,
        ),
        policy=policy,
        executor=_LaneImportExecutor(),
        component_store_root=target_store,
        component_reuse_roots=(legacy_components,),
    )

    assert manifest["status"] == "complete"
    execute_events = [
        (owner, component)
        for owner, component, event, _resource
        in corrected.events
        if event == "execute"
    ]
    assert execute_events == [(first_owner, "htr")]
    operational_attestations = tuple(
        (
            target_store.parent
            / "authenticated_component_imports"
        ).glob("*.json")
    )
    import_attestations = tuple(
        path
        for path in operational_attestations
        if json.loads(path.read_text(encoding="utf-8")).get(
            "schema_version"
        )
        == "production_role_neutral_authenticated_component_import_v3"
    )
    authentication_caches = tuple(
        path
        for path in operational_attestations
        if json.loads(path.read_text(encoding="utf-8")).get(
            "schema_version"
        )
        == "production_role_neutral_component_authentication_cache_v2"
    )
    assert len(import_attestations) == (
        len(plan.physical_scopes)
        * len(EXPECTED_COMPONENT_FAMILIES)
        - 1
    )
    expected_component_count = (
        len(plan.physical_scopes)
        * len(EXPECTED_COMPONENT_FAMILIES)
    )
    assert len(authentication_caches) == expected_component_count
    assert sum(
        event == "authenticate"
        for _owner, _component, event, _resource in corrected.events
    ) == expected_component_count
    for path in import_attestations:
        attestation = json.loads(path.read_text(encoding="utf-8"))
        assert attestation["schema_version"] == (
            "production_role_neutral_authenticated_component_import_v3"
        )
        assert attestation["source_authentication_mode"] == (
            "current_producer_deep_authentication_v1"
        )
        assert attestation[
            "source_authentication_cache_registration"
        ] is None
        assert attestation["source_payload_bytes_reauthenticated"] is True
        assert (
            attestation[
                "current_producer_semantic_authentication_count"
            ]
            == 1
        )
        assert (
            attestation["copied_tree_integrity_validation_count"]
            == 1
        )
        assert (
            attestation[
                "temporary_semantic_reauthentication_count"
            ]
            == 0
        )
        assert (
            attestation[
                "published_target_semantic_reauthentication_count"
            ]
            == 0
        )
    cached_owner_id = plan.physical_execution_order[1]
    cached_owner, cached_members = next(
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if owner.scope_id == cached_owner_id
    )
    cache_recorder = _ProducerRecorder()
    cached_result = _execute_one_owner(
        task=RoleNeutralPhysicalOwnerTask(
            plan=plan,
            physical_owner=cached_owner,
            logical_members=cached_members,
            component_parent=target_store / cached_owner_id,
            resource="cpu",
            resume=True,
            component_reuse_roots=(legacy_components,),
            component_import_attestation_root=(
                target_store.parent
                / "authenticated_component_imports"
            ),
        ),
        factories=cache_recorder.factories().as_mapping(),
        resume=True,
    )
    assert not [
        event
        for event in cache_recorder.events
        if event[2] == "authenticate"
    ]
    assert cached_result.execution_telemetry[
        "authentication_cache_hit_components"
    ] == list(EXPECTED_COMPONENT_FAMILIES)
    assert (
        legacy_components
        / first_owner
        / "htr"
        / ROLE_NEUTRAL_EXECUTION_MANIFEST
    ).read_bytes() == preserved_htr_terminal
    assert (
        target_store
        / first_owner
        / "htr"
        / ROLE_NEUTRAL_EXECUTION_MANIFEST
    ).is_file()


def test_cross_store_import_uses_protected_source_cache_without_replay(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    policy = RoleNeutralStage1ExecutionPolicy(
        resource_plan=_resource_plan(
            devices=("cpu",),
            cpu_budget=4,
        ),
        max_parallel_owners=1,
    )
    source_store = (tmp_path / "source-store" / "components").resolve()
    execute_and_publish_role_neutral_stage1(
        root=(tmp_path / "source-execution").resolve(),
        plan=plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=policy,
        executor=_RecordingExecutor(),
        component_store_root=source_store,
    )

    target_store = (tmp_path / "target-store" / "components").resolve()
    target_recorder = _ProducerRecorder()
    result = execute_and_publish_role_neutral_stage1(
        root=(tmp_path / "target-execution").resolve(),
        plan=plan,
        producer_factories=target_recorder.factories(),
        policy=policy,
        executor=_RecordingExecutor(),
        component_store_root=target_store,
        component_reuse_roots=(source_store,),
        component_stat_continuity_reuse_roots=(source_store,),
    )

    assert result["status"] == "complete"
    assert not [
        event
        for event in target_recorder.events
        if event[2] in {"execute", "authenticate"}
    ]
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            target_store.parent / "authenticated_component_imports"
        ).glob("*.json")
    ]
    imports = [
        record
        for record in records
        if record.get("schema_version")
        == "production_role_neutral_authenticated_component_import_v3"
    ]
    assert len(imports) == (
        len(plan.physical_scopes) * len(EXPECTED_COMPONENT_FAMILIES)
    )
    assert all(
        record["source_authentication_mode"]
        == "protected_cache_exact_stat_continuity_v1"
        and record["current_producer_semantic_authentication_count"] == 0
        and record["source_payload_bytes_reauthenticated"] is False
        and record["source_authentication_cache_registration"][
            "schema_version"
        ]
        == (
            "production_role_neutral_source_authentication_cache_registration_v1"
        )
        for record in imports
    )


def test_historical_full_authentication_reopens_by_exact_stat_continuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(gpu_ids=())
    owner, members = plan.physical_scope_groups[0]
    component_parent = (tmp_path / "components" / owner.scope_id).resolve()
    attestation_root = (
        tmp_path / "authenticated_component_imports"
    ).resolve()
    component_parent.mkdir(parents=True)
    attestation_root.mkdir()

    def write_closed(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
        value = {**body, "content_sha256": _sha(body)}
        _write_new_json(path, value)
        payload = path.read_bytes()
        return {
            "relative_path": path.relative_to(
                component_parent / str(body["component"])
            ).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "content_sha256": value["content_sha256"],
        }

    for component, families in EXPECTED_COMPONENT_FAMILIES.items():
        root = component_parent / component
        views_root = root / "logical_views"
        views_root.mkdir(parents=True)
        seal_registrations: dict[str, dict[str, Any]] = {}
        logical_views: list[dict[str, Any]] = []
        for family_index, family in enumerate(families):
            seal_path = root / f"fit_only_{family_index}.json"
            seal_registrations[family] = write_closed(
                seal_path,
                {
                    "schema_version": "test_prior_fit_seal_v1",
                    "component": component,
                    "physical_owner_scope_id": owner.scope_id,
                    "family": family,
                },
            )
            for member_index, member in enumerate(members):
                view_path = (
                    views_root
                    / f"{family_index}_{member_index}.json"
                )
                registration = write_closed(
                    view_path,
                    {
                        "schema_version": "test_prior_logical_view_v1",
                        "component": component,
                        "physical_owner_scope_id": owner.scope_id,
                        "logical_scope_id": member.scope_id,
                        "family": family,
                    },
                )
                logical_views.append(
                    {
                        **registration,
                        "logical_scope_id": member.scope_id,
                        **(
                            {}
                            if len(families) == 1
                            else {"family": family}
                        ),
                    }
                )
        group_request_body = {
            "schema_version": "test_prior_group_request_v1",
            "plan_scientific_content_sha256": (
                plan.scientific_content_sha256
            ),
            "physical_owner": owner.as_dict(),
            "logical_members": [
                member.as_dict() for member in members
            ],
            "fit_row_ids": list(owner.fit_row_ids),
            "canonical_group_seed": int(owner.scope_seed),
            "heldout_labels_supplied": False,
        }
        group_request = {
            **group_request_body,
            "content_sha256": _sha(group_request_body),
        }
        terminal_body = {
            "schema_version": "test_prior_component_terminal_v1",
            "status": "complete",
            "group_request": group_request,
            "fit_only_family_seal": (
                seal_registrations[families[0]]
                if len(families) == 1
                else None
            ),
            "fit_only_family_seals": (
                seal_registrations
                if len(families) > 1
                else None
            ),
            "logical_views": logical_views,
            "registered_heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "text_truncation_applied": False,
        }
        terminal = {
            **terminal_body,
            "content_sha256": _sha(terminal_body),
        }
        _write_new_json(
            root / ROLE_NEUTRAL_EXECUTION_MANIFEST,
            terminal,
        )
        tree_sha256 = (
            authenticated_role_neutral_component_tree_sha256(root)
        )
        attestation_body = {
            "schema_version": (
                "production_role_neutral_authenticated_component_import_v1"
            ),
            "physical_owner_scope_id": owner.scope_id,
            "component": component,
            "plan_scientific_content_sha256": (
                plan.scientific_content_sha256
            ),
            "source_components_root": str(component_parent),
            "source_terminal_content_sha256": terminal[
                "content_sha256"
            ],
            "source_tree_sha256": tree_sha256,
            "authentication_content_sha256": _sha(
                {
                    "owner": owner.scope_id,
                    "component": component,
                    "historical_full_authentication": True,
                }
            ),
            "current_producer_authenticated_source": True,
            "private_copy_not_link_or_reference": True,
            "current_producer_reauthenticated_temporary": True,
            "current_producer_reauthenticated_published_target": True,
            "source_tree_preserved": True,
        }
        _write_new_json(
            attestation_root
            / (
                _sha(
                    {
                        "physical_owner_scope_id": owner.scope_id,
                        "component": component,
                    }
                )
                + ".json"
            ),
            {
                **attestation_body,
                "content_sha256": _sha(attestation_body),
            },
        )

    task = RoleNeutralPhysicalOwnerTask(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component_parent=component_parent,
        resource="cpu",
        resume=True,
        component_import_attestation_root=attestation_root,
    )
    recorder = _ProducerRecorder()
    first = _execute_one_owner(
        task=task,
        factories=recorder.factories().as_mapping(),
        resume=True,
    )
    assert not [
        event for event in recorder.events if event[2] == "authenticate"
    ]
    assert first.execution_telemetry[
        "prior_authentication_continuity_components"
    ] == list(EXPECTED_COMPONENT_FAMILIES)
    caches = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in attestation_root.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8")).get(
            "schema_version"
        )
        == "production_role_neutral_component_authentication_cache_v2"
    ]
    assert len(caches) == len(EXPECTED_COMPONENT_FAMILIES)
    assert all(
        cache["authentication_basis"]
        == "prior_authenticated_component_import_v1_stat_continuity_v1"
        and cache["payload_bytes_reauthenticated"] is False
        for cache in caches
    )

    cached_recorder = _ProducerRecorder()
    cached = _execute_one_owner(
        task=task,
        factories=cached_recorder.factories().as_mapping(),
        resume=True,
    )
    assert not [
        event
        for event in cached_recorder.events
        if event[2] == "authenticate"
    ]
    assert cached.execution_telemetry[
        "authentication_cache_hit_components"
    ] == list(EXPECTED_COMPONENT_FAMILIES)

    import oci.inference.role_neutral_all_ten_binding as binding_module

    real_os_read = binding_module.os.read

    def reject_payload_reread(descriptor, size):
        opened_path = Path(
            binding_module.os.readlink(
                f"/proc/self/fd/{int(descriptor)}"
            )
        )
        if opened_path.name == ROLE_NEUTRAL_EXECUTION_MANIFEST:
            return real_os_read(descriptor, size)
        raise AssertionError(
            "unchanged component payload was redundantly reread"
        )

    monkeypatch.setattr(binding_module.os, "read", reject_payload_reread)
    for source in cached.sources:
        validate_authenticated_role_neutral_component_receipt(
            root=source.root,
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            receipt=source.receipt,
            expected_component=source.receipt.component,
        )

    terminal_path = (
        component_parent / "htr" / ROLE_NEUTRAL_EXECUTION_MANIFEST
    )
    terminal_path.write_bytes(terminal_path.read_bytes())
    with pytest.raises(
        ValueError,
        match="changed after its historical authentication",
    ):
        _prior_authenticated_component_receipt(
            attestation_root=attestation_root,
            component_root=component_parent / "htr",
            plan=plan,
            physical_owner_scope_id=owner.scope_id,
            component="htr",
        )


def test_component_store_namespace_is_science_narrow_and_finds_legacy_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(gpu_ids=())
    projection = {
        field: {"field": field}
        for field in (
            "dataset",
            "effective_stage1_config",
            "embedding_cache",
            "exact_inner_contract",
            "htr_input_nontruncation_audit",
            "htr_model",
            "query_config",
            "semantic_witness_scientific_config",
            "source_config",
            "split_registry_content_sha256",
            "stage1_scope_plan",
        )
    }
    prepared = SimpleNamespace(
        scientific_identity={
            "stage1_request_scientific_projection": projection,
        }
    )
    producer_scientific_identity = {
        "schema_version": "test_component_producer_science_v1",
        "architecture_profiles": {"all_ten": "unchanged"},
    }

    def integration_identity(integration):
        return {
            "producer_factories_builder": {
                "behavior_state": {
                    "state_policy": (
                        "explicit_closed_scientific_identity_v1"
                    ),
                    "scientific_identity": producer_scientific_identity,
                },
            },
            "stage2_handoff_publisher": {
                "content_sha256": integration.stage2_identity,
            },
        }

    monkeypatch.setattr(
        workflow_module,
        "_role_neutral_stage1_integration_identity",
        integration_identity,
    )
    workflow = object.__new__(
        workflow_module.ProductionAllEvidenceWorkflow
    )
    workflow.options = SimpleNamespace(
        scratch_root=(tmp_path / "scratch").resolve(),
        work_root=(tmp_path / "durable").resolve(),
    )
    first = workflow._stage1_component_store_root(
        prepared_context=prepared,
        plan=plan,
        integration=SimpleNamespace(stage2_identity="a" * 64),
    )
    second = workflow._stage1_component_store_root(
        prepared_context=prepared,
        plan=plan,
        integration=SimpleNamespace(stage2_identity="b" * 64),
    )
    assert first == second

    source_variant_projection = {
        **projection,
        "source_config": {"field": "byte-different-equivalent-profile"},
    }
    source_variant = workflow._stage1_component_store_root(
        prepared_context=SimpleNamespace(
            scientific_identity={
                "stage1_request_scientific_projection": (
                    source_variant_projection
                ),
            }
        ),
        plan=plan,
        integration=SimpleNamespace(stage2_identity="c" * 64),
    )
    assert source_variant == first

    effective_variant_projection = {
        **projection,
        "effective_stage1_config": {"field": "changed-science"},
    }
    effective_variant = workflow._stage1_component_store_root(
        prepared_context=SimpleNamespace(
            scientific_identity={
                "stage1_request_scientific_projection": (
                    effective_variant_projection
                ),
            }
        ),
        plan=plan,
        integration=SimpleNamespace(stage2_identity="d" * 64),
    )
    assert effective_variant != first

    namespace = first.parent.parent
    v2_store = namespace / ("e" * 64)
    v2_components = v2_store / "components"
    v2_components.mkdir(parents=True)
    v2_projection = {
        **projection,
        "source_config": {"field": "old-profile-byte-identity"},
    }
    v2_compatibility = {
        "schema_version": (
            "production_stage1_component_store_compatibility_v2"
        ),
        "prepared_stage1_component_input_projection": v2_projection,
        "prepared_stage1_component_input_projection_sha256": (
            _sha(v2_projection)
        ),
        "stage1_scope_plan_scientific_content_sha256": (
            plan.scientific_content_sha256
        ),
        "component_plan_namespace_identity": "v2",
        "component_producer_scientific_identity": (
            producer_scientific_identity
        ),
    }
    v2_body = {
        "schema_version": (
            "production_stage1_scientific_component_store_v2"
        ),
        "component_store_key": v2_store.name,
        "compatibility": v2_compatibility,
        "components_relative_path": "components",
        "successful_component_marker": "execution_manifest.json",
        "incomplete_attempts_preserved_for_recovery": True,
    }
    (
        v2_store
        / workflow_module.STAGE1_COMPONENT_STORE_MANIFEST
    ).write_text(
        json.dumps(
            {
                **v2_body,
                "content_sha256": _sha(v2_body),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    legacy_store = namespace / ("f" * 64)
    legacy_components = legacy_store / "components"
    legacy_components.mkdir(parents=True)
    legacy_compatibility = {
        "schema_version": (
            "production_stage1_component_store_compatibility_v1"
        ),
        "prepared_stage1_scientific_identity_sha256": "c" * 64,
        "stage1_scope_plan_scientific_content_sha256": (
            plan.scientific_content_sha256
        ),
        "component_plan_namespace_identity": "legacy",
        "component_producer_compatibility": {"legacy": True},
        "evidence_family_order": [],
        "resource_assignment_included": False,
        "cpu_budget_included": False,
        "owner_concurrency_included": False,
    }
    legacy_body = {
        "schema_version": (
            "production_stage1_scientific_component_store_v1"
        ),
        "component_store_key": legacy_store.name,
        "compatibility": legacy_compatibility,
        "components_relative_path": "components",
        "successful_component_marker": "execution_manifest.json",
        "incomplete_attempts_preserved_for_recovery": True,
    }
    (
        legacy_store
        / workflow_module.STAGE1_COMPONENT_STORE_MANIFEST
    ).write_text(
        json.dumps(
            {
                **legacy_body,
                "content_sha256": _sha(legacy_body),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    assert workflow._stage1_component_reuse_roots(
        component_store_root=first,
        plan=plan,
    ) == (
        v2_components.resolve(strict=True),
        legacy_components.resolve(strict=True),
    )
    assert workflow._stage1_component_reuse_roots(
        component_store_root=first,
        plan=plan,
        require_same_producer_identity=True,
    ) == (v2_components.resolve(strict=True),)
