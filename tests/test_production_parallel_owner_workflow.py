from __future__ import annotations

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
from oci.inference.portable_workflow_spec import DeploymentProfile
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
    execute_and_publish_role_neutral_stage1,
    validate_role_neutral_stage1_execution,
)
from oci.inference.role_neutral_all_ten_binding import (
    EXPECTED_COMPONENT_FAMILIES,
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
        executor=_RecordingExecutor(),
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
    import_attestations = tuple(
        (
            target_store.parent
            / "authenticated_component_imports"
        ).glob("*.json")
    )
    assert len(import_attestations) == (
        len(plan.physical_scopes)
        * len(EXPECTED_COMPONENT_FAMILIES)
        - 1
    )
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


def test_component_store_namespace_excludes_stage2_catalog_identity_and_finds_v1(
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

    namespace = first.parent.parent
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
    ) == (legacy_components.resolve(strict=True),)
