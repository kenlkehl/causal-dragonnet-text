from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from tests.resource_safety_test_support import resource_safety_policy

from oci.inference.portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
    ResourcePlan,
)
from oci.inference.portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
)
from oci.inference.production_stage1_legacy_scope_fragments import (
    build_role_neutral_fit_only_family_seal,
)
from oci.inference.production_stage1_role_neutral_coordinator import (
    _component_tree_sha256,
)
from oci.inference.production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    LocalThreadRoleNeutralPhysicalOwnerExecutor,
    NeuralQueryExecutionTopology,
    ROLE_NEUTRAL_COORDINATION_DIRECTORY,
    ROLE_NEUTRAL_EXECUTION_MANIFEST,
    RoleNeutralOperationalComponentReport,
    RoleNeutralPhysicalOwnerTask,
    RoleNeutralProducerFactories,
    RoleNeutralStage1ExecutionPolicy,
    _RoleNeutralStage1ParentSignal,
    _archive_stale_process_markers_for_resume,
    _execute_one_owner,
    execute_and_publish_role_neutral_stage1,
    validate_role_neutral_component_execution_intervals,
    validate_role_neutral_stage1_execution,
)
from oci.inference.production_stage1_scope_scheduler import (
    _WORKER_PROCESS_GROUP_MARKER_SCHEMA,
    _linux_process_start_time_ticks,
    build_canonical_stage1_scope_plan,
)
from tests.stage1_test_support import PHYSICAL_FIT_IDENTITY
from oci.inference.role_neutral_all_ten_binding import (
    AuthenticatedRoleNeutralComponentReceipt,
    EXPECTED_COMPONENT_FAMILIES,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _registry() -> dict:
    all_rows = tuple(range(25))
    folds = []
    for outer_fold in range(1, 6):
        heldout = tuple(range((outer_fold - 1) * 5, outer_fold * 5))
        fit = tuple(row for row in all_rows if row not in set(heldout))
        partitions = tuple(fit[index::5] for index in range(5))
        folds.append(
            {
                "outer_fold": outer_fold,
                "fit_row_ids": list(fit),
                "heldout_row_ids": list(heldout),
                "inner_folds": [
                    {
                        "inner_fold": index,
                        "fit_row_ids": [row for row in fit if row not in set(partition)],
                        "heldout_row_ids": list(partition),
                    }
                    for index, partition in enumerate(
                        partitions,
                        start=1,
                    )
                ],
            }
        )
    return {
        "dataset_row_count": len(all_rows),
        "outer_folds": folds,
    }


def _plan(
    *,
    gpu_ids: tuple[int, ...],
    registry_content_sha256: str = "a" * 64,
    physical_fit_identity=PHYSICAL_FIT_IDENTITY,
):
    return build_canonical_stage1_scope_plan(
        registry=_registry(),
        registry_content_sha256=registry_content_sha256,
        global_seed=42,
        physical_fit_identity=physical_fit_identity,
        gpu_ids=gpu_ids,
        review_rounds=2,
        initial_training_partitions=3,
        expected_outer_fold_count=5,
        expected_inner_fold_count=5,
    )


def _resource_plan(
    *,
    devices: tuple[str, ...],
    cpu_budget: int,
) -> ResourcePlan:
    gpus = tuple(
        GPUResource(
            device=device,
            uuid=f"uuid-{device}",
            total_memory_bytes=24 * 1024**3,
            free_memory_bytes=20 * 1024**3,
            utilization_percent=0.0,
        )
        for device in devices
        if device != "cpu"
    )
    return ResourcePlan(
        devices=devices,
        cpu_budget=cpu_budget,
        inventory=ResourceInventory(
            cpu_count=32,
            gpus=gpus,
        ),
        policy=devices,
        resource_performance_safety=resource_safety_policy(
            gpu_max_allocation_fraction=0.8,
            gpu_minimum_headroom_bytes=1024,
            minimum_multi_device_throughput_ratio=1.4,
            maximum_coordination_proof_overhead_ratio=0.25,
            maximum_ordinary_read_amplification=1.75,
            minimum_benchmark_repetitions_per_scope=3,
            read_counter_source="logical_read_bytes",
            fail_on_external_gpu_occupants=True,
        ),
    )


class _RecordingExecutor:
    def __init__(
        self,
        *,
        reverse_completion: bool = False,
        drop_last_result: bool = False,
    ):
        self.reverse_completion = reverse_completion
        self.drop_last_result = drop_last_result
        self.submitted: tuple[str, ...] = ()
        self.max_workers: int | None = None
        self.cpu_budget: int | None = None

    def execute(self, *, tasks, worker, max_workers, cpu_budget):
        self.submitted = tuple(task.physical_owner.scope_id for task in tasks)
        self.max_workers = int(max_workers)
        self.cpu_budget = int(cpu_budget)
        results = [worker(task) for task in tasks]
        if self.reverse_completion:
            results.reverse()
        if self.drop_last_result:
            results.pop()
        return tuple(results)


class _ProducerRecorder:
    def __init__(self):
        self.events: list[tuple[str, str, str, str]] = []

    def factory(self, expected_component: str):
        def bind(invocation):
            assert invocation.component == expected_component
            invocation.scientific_payload()
            owner_scope_id = invocation.physical_owner.scope_id
            self.events.append(
                (
                    owner_scope_id,
                    expected_component,
                    "factory",
                    invocation.resource,
                )
            )

            def execute():
                self.events.append(
                    (
                        owner_scope_id,
                        expected_component,
                        "execute",
                        invocation.resource,
                    )
                )
                invocation.output_root.mkdir(
                    parents=True,
                    exist_ok=False,
                )
                body = {
                    "schema_version": ("test_bound_role_neutral_component_terminal_v1"),
                    "plan_scientific_content_sha256": (invocation.plan.scientific_content_sha256),
                    "physical_owner_scope_id": owner_scope_id,
                    "component": expected_component,
                    "registered_heldout_labels_accessed": False,
                    "oracle_fields_accessed": False,
                    "text_truncation_applied": False,
                    "lossy_evidence_selection_applied": False,
                }
                terminal = {
                    **body,
                    "content_sha256": _sha(body),
                }
                (invocation.output_root / "execution_manifest.json").write_text(
                    json.dumps(
                        terminal,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                controls = invocation.htr_operational_controls
                if expected_component == "htr" and controls is not None:
                    reuse = controls.reuse_tokenizer_and_chunk_plans
                    operational_body = {
                        "schema_version": (
                            "production_role_neutral_htr_operational_attestation_v2"
                        ),
                        "controls": controls.as_dict(),
                        "scientific_training_batch_size": (
                            controls.training_batch_size
                        ),
                        "training_batch_override_applied": False,
                        "scientific_sentence_encoder_batch_size": (
                            controls.sentence_encoder_batch_size
                        ),
                        "effective_sentence_encoder_batch_size": (
                            controls.sentence_encoder_batch_size
                        ),
                        "fit_reusable_plan": (
                            {
                                "content_sha256": "a" * 64,
                                "unique_note_count": 1,
                                "unique_chunk_count": 1,
                                "parallel_plan_task_count": (
                                    1 if controls.data_loader_workers else 0
                                ),
                                "parallel_plan_thread_count": (
                                    1 if controls.data_loader_workers else 0
                                ),
                                "positive_data_loader_workers_exercised": (
                                    controls.data_loader_workers > 0
                                ),
                            }
                            if reuse
                            else None
                        ),
                        "exact_heldout_reusable_plan": (
                            {
                                "content_sha256": "b" * 64,
                                "unique_note_count": 1,
                                "unique_chunk_count": 1,
                                "parallel_plan_task_count": (
                                    1 if controls.data_loader_workers else 0
                                ),
                                "parallel_plan_thread_count": (
                                    1 if controls.data_loader_workers else 0
                                ),
                                "positive_data_loader_workers_exercised": (
                                    controls.data_loader_workers > 0
                                ),
                            }
                            if reuse
                            else None
                        ),
                        "cache_capacities_nonbinding": True,
                        "positive_data_loader_workers_exercised": True,
                        "replay_comparison_policy": (
                            "allclose_and_exact_discrete_state_v1"
                        ),
                        "replay_relative_tolerance": 1e-4,
                        "replay_absolute_tolerance": 1e-5,
                        "operational_predictions_within_declared_tolerance_of_scientific_replay": True,
                        "complete_artifact_equality_decided_by_benchmark": True,
                        "raw_text_persisted_in_operational_attestation": False,
                        "semantic_truncation_applied": False,
                    }
                    return RoleNeutralOperationalComponentReport(
                        component="htr",
                        attestation={
                            **operational_body,
                            "content_sha256": _sha(operational_body),
                        },
                    )
                return None

            def authenticate():
                self.events.append(
                    (
                        owner_scope_id,
                        expected_component,
                        "authenticate",
                        invocation.resource,
                    )
                )
                terminal = json.loads(
                    (invocation.output_root / "execution_manifest.json").read_text(encoding="utf-8")
                )
                if (
                    terminal.get("plan_scientific_content_sha256")
                    != invocation.plan.scientific_content_sha256
                    or terminal.get("physical_owner_scope_id")
                    != owner_scope_id
                    or terminal.get("component")
                    != expected_component
                ):
                    raise ValueError(
                        "component terminal belongs to another request identity"
                    )
                seals = {}
                views = {}
                for family in EXPECTED_COMPONENT_FAMILIES[expected_component]:
                    payload = {
                        "schema_version": ("native_stage1_family_concept_evidence_v1"),
                        "family": family,
                        "architecture_evidence": [
                            {
                                "kind": "orchestration_contract_atom",
                                "physical_owner_scope_id": owner_scope_id,
                                "complete_family_payload": True,
                            }
                        ],
                    }
                    seals[family] = build_role_neutral_fit_only_family_seal(
                        plan=invocation.plan,
                        physical_owner_scope_id=owner_scope_id,
                        family=family,
                        evidence_payload=payload,
                        producer_identity_sha256=_sha(
                            {
                                "component": expected_component,
                                "family": family,
                                "producer": ("bound_producer_contract"),
                            }
                        ),
                        configuration_identity_sha256=_sha(
                            {
                                "component": expected_component,
                                "family": family,
                                "configuration": "explicit",
                            }
                        ),
                        fit_state_artifact_sha256=_sha(
                            {
                                "owner": owner_scope_id,
                                "family": family,
                                "fit_state": "complete",
                            }
                        ),
                    )
                    views[family] = {
                        member.scope_id: _sha(
                            {
                                "owner": owner_scope_id,
                                "logical_scope": member.scope_id,
                                "purpose": member.scope_kind,
                                "component": expected_component,
                                "family": family,
                            }
                        )
                        for member in invocation.logical_members
                    }
                return AuthenticatedRoleNeutralComponentReceipt.create(
                    plan=invocation.plan,
                    physical_owner_scope_id=owner_scope_id,
                    component=expected_component,
                    family_fit_seals=seals,
                    family_logical_view_content_sha256=views,
                    source_terminal_content_sha256=terminal["content_sha256"],
                    source_tree_sha256=_component_tree_sha256(invocation.output_root),
                )

            return BoundRoleNeutralComponentProducer(
                execute=execute,
                authenticate=authenticate,
            )

        return bind

    def factories(self) -> RoleNeutralProducerFactories:
        return RoleNeutralProducerFactories(
            bow=self.factory("bow"),
            htr=self.factory("htr"),
            matched_pair=self.factory("matched_pair"),
            embeddings=self.factory("embeddings"),
            tfidf=self.factory("tfidf"),
            neural_query=self.factory("neural_query"),
        )


def test_accelerator_owner_emits_six_direct_closed_component_intervals(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    owner, members = plan.physical_scope_groups[0]
    task = RoleNeutralPhysicalOwnerTask(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component_parent=(tmp_path / "interval-components").resolve(),
        resource="cuda:3",
        neural_query_execution_topology=(
            NeuralQueryExecutionTopology(
                devices=("cuda:3", "cuda:8"),
            )
        ),
    )
    result = _execute_one_owner(
        task=task,
        factories=_ProducerRecorder().factories().as_mapping(),
    )
    intervals = validate_role_neutral_component_execution_intervals(
        execution_telemetry=result.execution_telemetry,
        expected_physical_owner_scope_id=owner.scope_id,
        expected_primary_resource="cuda:3",
        expected_neural_query_resources=("cuda:3", "cuda:8"),
    )

    assert tuple(row["component"] for row in intervals) == tuple(
        EXPECTED_COMPONENT_FAMILIES
    )
    assert tuple(row["lane_kind"] for row in intervals) == (
        "cpu",
        "gpu",
        "gpu",
        "cpu",
        "cpu",
        "gpu",
    )
    assert tuple(row["resource_ids"] for row in intervals) == (
        ["host_cpu"],
        ["cuda:3"],
        ["cuda:3"],
        ["host_cpu"],
        ["host_cpu"],
        ["cuda:3", "cuda:8"],
    )
    assert all(
        row["physical_owner_scope_id"] == owner.scope_id
        and row["timestamps_measured_directly"] is True
        and row["status"] == "completed"
        and row["finished_monotonic_ns"] > row["started_monotonic_ns"]
        for row in intervals
    )
    assert all(
        left["finished_monotonic_ns"]
        <= right["started_monotonic_ns"]
        for left, right in zip(intervals, intervals[1:])
    )

    for mutation in ("missing", "reordered", "tampered"):
        changed = copy.deepcopy(dict(result.execution_telemetry))
        rows = changed["component_execution_intervals"]
        if mutation == "missing":
            rows.pop()
        elif mutation == "reordered":
            rows[0], rows[1] = rows[1], rows[0]
        else:
            rows[-1]["resource_ids"] = ["cuda:8", "cuda:3"]
        with pytest.raises(ValueError, match="component execution"):
            validate_role_neutral_component_execution_intervals(
                execution_telemetry=changed,
                expected_physical_owner_scope_id=owner.scope_id,
                expected_primary_resource="cuda:3",
                expected_neural_query_resources=("cuda:3", "cuda:8"),
            )


def test_owner_resume_reuses_complete_component_prefix(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    owner, members = plan.physical_scope_groups[0]
    task = RoleNeutralPhysicalOwnerTask(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component_parent=(
            tmp_path
            / "attempt"
            / "role_neutral_stage1_execution"
            / "components"
            / owner.scope_id
        ).resolve(),
        resource="cpu",
    )
    first = _ProducerRecorder()
    first_factories = first.factories()

    def failing_neural(invocation):
        bound = first_factories.neural_query(invocation)

        def fail():
            invocation.output_root.mkdir(parents=True)
            (invocation.output_root / "partial.bin").write_bytes(
                b"interrupted"
            )
            raise RuntimeError("simulated neural interruption")

        return BoundRoleNeutralComponentProducer(
            execute=fail,
            authenticate=bound.authenticate,
        )

    failed = RoleNeutralProducerFactories(
        bow=first_factories.bow,
        htr=first_factories.htr,
        matched_pair=first_factories.matched_pair,
        embeddings=first_factories.embeddings,
        tfidf=first_factories.tfidf,
        neural_query=failing_neural,
    )
    with pytest.raises(RuntimeError, match="neural interruption"):
        _execute_one_owner(
            task=task,
            factories=failed.as_mapping(),
        )

    resumed = _ProducerRecorder()
    resumed_factories = resumed.factories()

    def bind_fresh_neural(invocation):
        assert not invocation.output_root.exists()
        return resumed_factories.neural_query(invocation)

    result = _execute_one_owner(
        task=replace(task, resume=True),
        factories=RoleNeutralProducerFactories(
            bow=resumed_factories.bow,
            htr=resumed_factories.htr,
            matched_pair=resumed_factories.matched_pair,
            embeddings=resumed_factories.embeddings,
            tfidf=resumed_factories.tfidf,
            neural_query=bind_fresh_neural,
        ).as_mapping(),
    )
    intervals = validate_role_neutral_component_execution_intervals(
        execution_telemetry=result.execution_telemetry,
        expected_physical_owner_scope_id=owner.scope_id,
        expected_primary_resource="cpu",
        expected_neural_query_resources=("cpu",),
    )

    assert [row["status"] for row in intervals] == [
        "resumed",
        "resumed",
        "resumed",
        "resumed",
        "resumed",
        "completed",
    ]
    assert [
        event
        for _owner, component, event, _resource in resumed.events
        if component != "neural_query"
    ] == ["factory", "authenticate"] * 5
    assert [
        event
        for _owner, component, event, _resource in resumed.events
        if component == "neural_query"
    ] == ["factory", "execute", "authenticate"]
    interrupted = tuple(
        (
            tmp_path
            / "attempt"
            / "interrupted_role_neutral_components"
            / owner.scope_id
        ).glob("neural_query.*")
    )
    assert len(interrupted) == 1
    assert (interrupted[0] / "partial.bin").read_bytes() == b"interrupted"


def test_owner_resume_rejects_completed_components_from_another_identity(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    owner, members = plan.physical_scope_groups[0]
    component_parent = (
        tmp_path
        / "attempt"
        / "role_neutral_stage1_execution"
        / "components"
        / owner.scope_id
    ).resolve()
    _execute_one_owner(
        task=RoleNeutralPhysicalOwnerTask(
            plan=plan,
            physical_owner=owner,
            logical_members=members,
            component_parent=component_parent,
            resource="cpu",
        ),
        factories=_ProducerRecorder().factories().as_mapping(),
    )
    original_terminal = (
        component_parent / "bow" / "execution_manifest.json"
    ).read_bytes()

    drifted_plans = (
        _plan(
            gpu_ids=(),
            registry_content_sha256="b" * 64,
        ),
        _plan(
            gpu_ids=(),
            physical_fit_identity=replace(
                PHYSICAL_FIT_IDENTITY,
                architecture_identity="b" * 64,
            ),
        ),
    )
    for drifted_plan in drifted_plans:
        drifted_owner, drifted_members = (
            drifted_plan.physical_scope_groups[0]
        )
        recorder = _ProducerRecorder()
        with pytest.raises(
            ValueError,
            match="another request identity",
        ):
            _execute_one_owner(
                task=RoleNeutralPhysicalOwnerTask(
                    plan=drifted_plan,
                    physical_owner=drifted_owner,
                    logical_members=drifted_members,
                    component_parent=component_parent,
                    resource="cpu",
                    resume=True,
                ),
                factories=recorder.factories().as_mapping(),
            )
        assert [
            event
            for _owner, _component, event, _resource in recorder.events
        ] == ["factory", "authenticate"]
        assert (
            component_parent / "bow" / "execution_manifest.json"
        ).read_bytes() == original_terminal


def test_stage1_resume_finishes_partial_execution_root(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    root = (tmp_path / "partial-execution").resolve()
    component_root = root / "components"
    component_root.mkdir(parents=True)
    owner, members = plan.physical_scope_groups[0]
    task = RoleNeutralPhysicalOwnerTask(
        plan=plan,
        physical_owner=owner,
        logical_members=members,
        component_parent=component_root / owner.scope_id,
        resource="cpu",
    )
    first = _ProducerRecorder()
    first_factories = first.factories()

    def failing_neural(invocation):
        bound = first_factories.neural_query(invocation)
        return BoundRoleNeutralComponentProducer(
            execute=lambda: (_ for _ in ()).throw(
                RuntimeError("simulated interruption")
            ),
            authenticate=bound.authenticate,
        )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _execute_one_owner(
            task=task,
            factories=RoleNeutralProducerFactories(
                bow=first_factories.bow,
                htr=first_factories.htr,
                matched_pair=first_factories.matched_pair,
                embeddings=first_factories.embeddings,
                tfidf=first_factories.tfidf,
                neural_query=failing_neural,
            ).as_mapping(),
        )

    resumed = _ProducerRecorder()
    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=resumed.factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cpu",),
                cpu_budget=4,
            ),
            max_parallel_owners=1,
        ),
        executor=_RecordingExecutor(),
        resume=True,
    )

    assert manifest["status"] == "complete"
    first_owner_events = [
        (component, event)
        for event_owner, component, event, _resource in resumed.events
        if event_owner == owner.scope_id
    ]
    assert first_owner_events == [
        *((component, event) for component in tuple(EXPECTED_COMPONENT_FAMILIES)[:-1]
          for event in ("factory", "authenticate")),
        ("neural_query", "factory"),
        ("neural_query", "execute"),
        ("neural_query", "authenticate"),
    ]


def test_resume_archives_only_process_markers_confirmed_not_live(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "execution").resolve()
    session_root = root / ".persistent-owner-execution-session"
    session_root.mkdir(parents=True)

    def write_marker(path: Path, pid: int, start_time: int) -> None:
        body = {
            "schema_version": _WORKER_PROCESS_GROUP_MARKER_SCHEMA,
            "pid": pid,
            "process_group_id": pid,
            "process_start_time_ticks": start_time,
        }
        path.write_text(
            json.dumps(
                {**body, "content_sha256": _sha(body)},
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    dead_pid = 2**30
    write_marker(session_root / "process-group-slot-0.json", dead_pid, 1)
    _archive_stale_process_markers_for_resume(root)
    assert not session_root.exists()
    archived = tuple(
        (tmp_path / "interrupted_role_neutral_process_markers").glob(
            ".persistent-owner-execution-session.*"
        )
    )
    assert len(archived) == 1

    session_root.mkdir()
    current_start = _linux_process_start_time_ticks(os.getpid())
    assert current_start is not None
    live_marker = session_root / "process-group-slot-0.json"
    write_marker(live_marker, os.getpid(), current_start)
    with pytest.raises(RuntimeError, match="worker is live"):
        _archive_stale_process_markers_for_resume(root)
    assert live_marker.is_file()


def test_parent_sigterm_interrupts_persistent_stage1_session(
    tmp_path: Path,
) -> None:
    class Session:
        interrupted = False
        exited = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback_value):
            self.exited = True
            return False

        def interrupt(self):
            self.interrupted = True

        def execute(self, **_kwargs):
            signal.raise_signal(signal.SIGTERM)
            raise AssertionError("SIGTERM handler did not interrupt execution")

    class Executor:
        process_isolated_physical_owners = True

        def __init__(self):
            self.session = Session()

        def execute(self, **_kwargs):
            raise AssertionError("unscoped executor path was used")

        def open_session(self, **_kwargs):
            return self.session

    executor = Executor()
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(
        _RoleNeutralStage1ParentSignal,
        match="received signal",
    ):
        execute_and_publish_role_neutral_stage1(
            root=(tmp_path / "sigterm-execution").resolve(),
            plan=_plan(gpu_ids=()),
            producer_factories=_ProducerRecorder().factories(),
            policy=RoleNeutralStage1ExecutionPolicy(
                resource_plan=_resource_plan(
                    devices=("cpu",),
                    cpu_budget=2,
                ),
                max_parallel_owners=1,
            ),
            executor=executor,
        )
    assert executor.session.interrupted is True
    assert executor.session.exited is True
    assert signal.getsignal(signal.SIGTERM) is previous


def test_parent_sigterm_reaches_fresh_executor_cleanup(
    tmp_path: Path,
) -> None:
    class Executor:
        process_isolated_physical_owners = True
        cleanup_reached = False

        def execute(self, **_kwargs):
            try:
                signal.raise_signal(signal.SIGTERM)
                raise AssertionError(
                    "SIGTERM handler did not interrupt execution"
                )
            finally:
                self.cleanup_reached = True

    executor = Executor()
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(
        _RoleNeutralStage1ParentSignal,
        match="received signal",
    ):
        execute_and_publish_role_neutral_stage1(
            root=(tmp_path / "fresh-sigterm-execution").resolve(),
            plan=_plan(gpu_ids=()),
            producer_factories=_ProducerRecorder().factories(),
            policy=RoleNeutralStage1ExecutionPolicy(
                resource_plan=_resource_plan(
                    devices=("cpu",),
                    cpu_budget=2,
                ),
                max_parallel_owners=1,
            ),
            executor=executor,
        )
    assert executor.cleanup_reached is True
    assert signal.getsignal(signal.SIGTERM) is previous


def test_executes_derived_physical_owners_once_and_publishes_all_ten(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=(3, 8))
    recorder = _ProducerRecorder()
    executor = _RecordingExecutor(reverse_completion=True)
    root = (tmp_path / "execution").resolve()
    manifest = execute_and_publish_role_neutral_stage1(
        root=root,
        plan=plan,
        producer_factories=recorder.factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cuda:3", "cuda:8"),
                cpu_budget=8,
            ),
            max_parallel_owners=4,
        ),
        executor=executor,
    )

    assert len(plan.physical_scopes) == 35
    assert len(plan.scopes) == 40
    assert manifest["physical_fit_count"] == len(plan.physical_scopes)
    assert manifest["logical_scope_count"] == len(plan.scopes)
    assert manifest["deduplicated_fit_count"] == 5
    assert manifest["every_component_executed_and_authenticated_once_per_owner"] is True
    assert manifest["productive_compute_canary_completed"] is False
    assert manifest["selected_canary_replica_adopted_as_production"] is False
    assert manifest["compute_canary_scientific_equality"] is None
    assert not (root / "compute_canary_attestation.json").exists()
    execution_attestation = json.loads(
        (root / "execution_attestation.json").read_text(encoding="utf-8")
    )
    assert execution_attestation["max_parallel_owners"] == 4
    assert execution_attestation["effective_max_parallel_owners"] == 4
    assert execution_attestation["owner_cpu_budget"] == 2
    assert (
        execution_attestation["effective_owner_concurrency_policy"]
        == "configured_topology_capacity_v1"
    )
    assert execution_attestation["compute_canary"] is None
    assert execution_attestation["compute_canary_replica_execution_count"] == 0
    assert (
        execution_attestation[
            "compute_canary_additional_physical_execution_count"
        ]
        == 0
    )
    assert executor.submitted == plan.physical_execution_order
    assert executor.max_workers == 4
    assert executor.cpu_budget == 8
    assert (
        validate_role_neutral_stage1_execution(
            root=root,
            plan=plan,
        )
        == manifest
    )

    counts = Counter(
        (owner, component, event) for owner, component, event, _resource in recorder.events
    )
    assert set(counts.values()) == {1}
    assert len(counts) == (len(plan.physical_scopes) * len(EXPECTED_COMPONENT_FAMILIES) * 3)
    for owner in executor.submitted:
        owner_events = [
            (component, event)
            for event_owner, component, event, _resource in recorder.events
            if event_owner == owner
        ]
        assert owner_events == [
            (component, event)
            for component in EXPECTED_COMPONENT_FAMILIES
            for event in ("factory", "execute", "authenticate")
        ]


def test_local_single_node_executor_runs_the_complete_physical_plan(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    recorder = _ProducerRecorder()
    manifest = execute_and_publish_role_neutral_stage1(
        root=(tmp_path / "local_thread_execution").resolve(),
        plan=plan,
        producer_factories=recorder.factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cpu",),
                cpu_budget=4,
            ),
            max_parallel_owners=3,
        ),
        executor=LocalThreadRoleNeutralPhysicalOwnerExecutor(
            thread_name_prefix="test-stage1-owner",
        ),
    )

    assert manifest["physical_fit_count"] == len(plan.physical_scopes)
    assert manifest["logical_scope_count"] == len(plan.scopes)
    assert len(recorder.events) == (
        len(plan.physical_scopes) * len(EXPECTED_COMPONENT_FAMILIES) * 3
    )


def test_scientific_identity_is_path_device_worker_and_completion_neutral(
    tmp_path: Path,
):
    first_plan = _plan(gpu_ids=(3, 8))
    second_plan = _plan(gpu_ids=(17,))
    assert first_plan.scientific_content_sha256 == second_plan.scientific_content_sha256
    assert first_plan.content_sha256 != second_plan.content_sha256

    first = execute_and_publish_role_neutral_stage1(
        root=(tmp_path / "first_execution").resolve(),
        plan=first_plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cuda:3", "cuda:8"),
                cpu_budget=8,
            ),
            max_parallel_owners=1,
        ),
        executor=_RecordingExecutor(reverse_completion=False),
    )
    second = execute_and_publish_role_neutral_stage1(
        root=(tmp_path / "relocated_second_execution").resolve(),
        plan=second_plan,
        producer_factories=_ProducerRecorder().factories(),
        policy=RoleNeutralStage1ExecutionPolicy(
            resource_plan=_resource_plan(
                devices=("cuda:17",),
                cpu_budget=12,
            ),
            max_parallel_owners=7,
        ),
        executor=_RecordingExecutor(reverse_completion=True),
    )
    assert first["scientific_identity"] == second["scientific_identity"]
    assert first["content_sha256"] != second["content_sha256"]


def test_incomplete_executor_results_fail_before_coordination_publication(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=())
    root = (tmp_path / "incomplete_execution").resolve()
    with pytest.raises(
        ValueError,
        match="missing, duplicate, or extra owners",
    ):
        execute_and_publish_role_neutral_stage1(
            root=root,
            plan=plan,
            producer_factories=_ProducerRecorder().factories(),
            policy=RoleNeutralStage1ExecutionPolicy(
                resource_plan=_resource_plan(
                    devices=("cpu",),
                    cpu_budget=4,
                ),
                max_parallel_owners=2,
            ),
            executor=_RecordingExecutor(drop_last_result=True),
        )
    assert not (root / ROLE_NEUTRAL_COORDINATION_DIRECTORY).exists()
    assert not (root / ROLE_NEUTRAL_EXECUTION_MANIFEST).exists()


def test_factory_contract_is_closed_before_any_output_is_created(
    tmp_path: Path,
):
    plan = _plan(gpu_ids=())
    recorder = _ProducerRecorder()
    valid = recorder.factories()
    invalid = RoleNeutralProducerFactories(
        bow=valid.bow,
        htr=valid.htr,
        matched_pair=valid.matched_pair,
        embeddings=valid.embeddings,
        tfidf=valid.tfidf,
        neural_query=None,  # type: ignore[arg-type]
    )
    root = (tmp_path / "invalid_factories").resolve()
    with pytest.raises(TypeError, match="every role-neutral producer"):
        execute_and_publish_role_neutral_stage1(
            root=root,
            plan=plan,
            producer_factories=invalid,
            policy=RoleNeutralStage1ExecutionPolicy(
                resource_plan=_resource_plan(
                    devices=("cpu",),
                    cpu_budget=2,
                ),
                max_parallel_owners=1,
            ),
            executor=_RecordingExecutor(),
        )
    assert not root.exists()
