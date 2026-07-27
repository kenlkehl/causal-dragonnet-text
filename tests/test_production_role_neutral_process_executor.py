from __future__ import annotations

import dataclasses
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import oci.inference.prepared_stage1_context as prepared_context_module
from oci.inference.production_role_neutral_process_executor import (
    ProcessIsolatedRoleNeutralPhysicalOwnerExecutor,
    _execute_production_role_neutral_owner,
)
from oci.inference.production_role_neutral_persistent_executor import (
    PERSISTENT_ROLE_NEUTRAL_WORKER_MODE,
    PersistentSpawnRoleNeutralPhysicalOwnerExecutor,
)
from oci.inference.production_stage1_role_neutral_execution import (
    RoleNeutralPhysicalOwnerTask,
    _execute_one_owner,
    _freshly_reauthenticate_owner_result,
)
from oci.inference.role_neutral_all_ten_binding import (
    EXPECTED_COMPONENT_FAMILIES,
    validate_authenticated_role_neutral_component_receipt,
)

from tests.test_production_stage1_role_neutral_execution import (
    _ProducerRecorder,
    _plan,
)


def _fake_spawn_owner_worker(*, task, worker_parameters):
    if worker_parameters != {"test_token": "mixed-seed-isolation"}:
        raise ValueError("test worker parameters changed")
    probe = {
        "pid": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "native_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "python_random": random.random(),
        "numpy_random": float(np.random.random()),
        "torch_random": float(torch.rand(1).item()),
    }
    result = _execute_one_owner(
        task=task,
        factories=_ProducerRecorder().factories().as_mapping(),
    )
    assert result.execution_telemetry is not None
    return dataclasses.replace(
        result,
        execution_telemetry={
            **dict(result.execution_telemetry),
            **probe,
        },
    )


def _must_not_execute_in_parent(_task):
    raise AssertionError("process executor invoked the parent worker closure")


def _tasks(plan, parent: Path, count: int = 2):
    parent.mkdir(parents=True, exist_ok=False)
    groups = {
        owner.scope_id: (owner, members)
        for owner, members in plan.physical_scope_groups
    }
    return tuple(
        RoleNeutralPhysicalOwnerTask(
            plan=plan,
            physical_owner=groups[owner.scope_id][0],
            logical_members=groups[owner.scope_id][1],
            component_parent=(parent / owner.scope_id).resolve(),
            resource="cpu",
        )
        for owner in plan.physical_scopes[:count]
    )


def _scientific_by_owner(results):
    return {
        result.physical_owner_scope_id: tuple(
            source.receipt.scientific_dict() for source in result.sources
        )
        for result in results
    }


def _assert_reopened_cpu_component_intervals(report, *, owner_scope_id):
    assert report["schema_version"] == (
        "production_role_neutral_component_operational_reports_v2"
    )
    intervals = report["component_execution_intervals"]
    assert len(intervals) == len(EXPECTED_COMPONENT_FAMILIES)
    assert tuple(row["component"] for row in intervals) == tuple(
        EXPECTED_COMPONENT_FAMILIES
    )
    assert all(
        row["physical_owner_scope_id"] == owner_scope_id
        and row["lane_kind"] == "cpu"
        and row["resource_ids"] == ["host_cpu"]
        and row["timestamps_measured_directly"] is True
        and row["finished_monotonic_ns"] > row["started_monotonic_ns"]
        for row in intervals
    )


def test_fresh_production_worker_reconstructs_sealed_context_without_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(gpu_ids=())
    task = _tasks(
        plan,
        (tmp_path / "fresh_context_components").resolve(),
        count=1,
    )[0]
    factories = _ProducerRecorder().factories()
    reconstruct_budgets: list[int] = []

    class _Context:
        def reconstruct(self, *, slot_cpu_budget):
            reconstruct_budgets.append(int(slot_cpu_budget))
            return (
                type("_Prepared", (), {"stage1_scope_plan": plan})(),
                factories,
            )

    monkeypatch.setattr(
        prepared_context_module,
        "load_prepared_stage1_context",
        lambda _path: _Context(),
    )
    from oci.inference.production_stage1_bundle import (
        ProductionStage1BundleBuilder,
    )

    monkeypatch.setattr(
        ProductionStage1BundleBuilder,
        "prepare",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("fresh worker reran monolithic prepare")
        ),
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "3")

    result = _execute_production_role_neutral_owner(
        task=task,
        worker_parameters={
            "prepared_context_manifest_path": str(
                (tmp_path / "prepared_stage1_context_manifest.json").resolve()
            )
        },
    )

    assert reconstruct_budgets == [3]
    assert result.physical_owner_scope_id == task.physical_owner.scope_id
    assert len(result.sources) == 6


def test_mixed_owner_seeds_are_spawn_isolated_and_schedule_equal(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    sequential_tasks = _tasks(
        plan,
        (tmp_path / "sequential_components").resolve(),
    )
    concurrent_tasks = _tasks(
        plan,
        (tmp_path / "concurrent_components").resolve(),
    )
    executor = ProcessIsolatedRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=2,
        worker_target=f"{__name__}:_fake_spawn_owner_worker",
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
        poll_interval_seconds=0.01,
    )

    sequential = tuple(
        executor.execute(
            tasks=sequential_tasks,
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=2,
        )
    )
    concurrent = tuple(
        executor.execute(
            tasks=concurrent_tasks,
            worker=_must_not_execute_in_parent,
            max_workers=2,
            cpu_budget=2,
        )
    )

    assert _scientific_by_owner(sequential) == _scientific_by_owner(concurrent)
    sequential_by_owner = {
        result.physical_owner_scope_id: result for result in sequential
    }
    concurrent_by_owner = {
        result.physical_owner_scope_id: result for result in concurrent
    }
    probes = []
    for task in sequential_tasks:
        owner = task.physical_owner.scope_id
        first = sequential_by_owner[owner].execution_telemetry
        second = concurrent_by_owner[owner].execution_telemetry
        assert first is not None and second is not None
        first_report = first["worker_report"]
        second_report = second["worker_report"]
        for report in (first_report, second_report):
            assert report["pid"] != os.getpid()
            assert report["python_hash_seed"] == str(
                task.physical_owner.scope_seed
            )
            _assert_reopened_cpu_component_intervals(
                report,
                owner_scope_id=owner,
            )
        assert set(first_report["native_thread_environment"].values()) == {
            str(first["native_threads"])
        }
        assert set(second_report["native_thread_environment"].values()) == {
            str(second["native_threads"])
        }
        assert (
            first_report["python_random"],
            first_report["numpy_random"],
            first_report["torch_random"],
        ) == (
            second_report["python_random"],
            second_report["numpy_random"],
            second_report["torch_random"],
        )
        probes.append(
            (
                first_report["python_random"],
                first_report["numpy_random"],
                first_report["torch_random"],
            )
        )
        _freshly_reauthenticate_owner_result(
            task=task,
            result=sequential_by_owner[owner],
        )
    assert len(set(probes)) == len(probes)
    assert not tuple(
        (tmp_path / "sequential_components").glob(".process-group-*.json")
    )
    assert not tuple(
        (tmp_path / "concurrent_components").glob(".process-group-*.json")
    )


def test_persistent_slots_reseed_mixed_owners_and_cleanup(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    sequential_tasks = _tasks(
        plan,
        (tmp_path / "persistent_sequential").resolve(),
    )
    concurrent_tasks = _tasks(
        plan,
        (tmp_path / "persistent_concurrent").resolve(),
    )
    executor = PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=2,
        startup_timeout_seconds=30.0,
        worker_target=f"{__name__}:_fake_spawn_owner_worker",
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
        poll_interval_seconds=0.01,
    )

    sequential = tuple(
        executor.execute(
            tasks=sequential_tasks,
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=2,
        )
    )
    concurrent = tuple(
        executor.execute(
            tasks=concurrent_tasks,
            worker=_must_not_execute_in_parent,
            max_workers=2,
            cpu_budget=2,
        )
    )

    assert _scientific_by_owner(sequential) == _scientific_by_owner(concurrent)
    sequential_by_owner = {
        result.physical_owner_scope_id: result for result in sequential
    }
    concurrent_by_owner = {
        result.physical_owner_scope_id: result for result in concurrent
    }
    assert len(
        {
            result.execution_telemetry["pid"]
            for result in sequential
        }
    ) == 1
    assert len(
        {
            result.execution_telemetry["pid"]
            for result in concurrent
        }
    ) == 2
    for task in sequential_tasks:
        owner = task.physical_owner.scope_id
        first = sequential_by_owner[owner].execution_telemetry
        second = concurrent_by_owner[owner].execution_telemetry
        assert first["worker_lifecycle_mode"] == (
            PERSISTENT_ROLE_NEUTRAL_WORKER_MODE
        )
        assert first["host_cpu_budget"] == 2
        assert first["slot_cpu_budget"] == 2
        assert second["host_cpu_budget"] == 2
        assert second["slot_cpu_budget"] == 1
        first_report = first["worker_report"]
        second_report = second["worker_report"]
        _assert_reopened_cpu_component_intervals(
            first_report,
            owner_scope_id=owner,
        )
        _assert_reopened_cpu_component_intervals(
            second_report,
            owner_scope_id=owner,
        )
        assert (
            first_report["python_random"],
            first_report["numpy_random"],
            first_report["torch_random"],
        ) == (
            second_report["python_random"],
            second_report["numpy_random"],
            second_report["torch_random"],
        )
        _freshly_reauthenticate_owner_result(
            task=task,
            result=sequential_by_owner[owner],
        )
    assert not tuple(
        (tmp_path / "persistent_sequential").glob(
            ".persistent-process-group-*.json"
        )
    )
    assert not tuple(
        (tmp_path / "persistent_concurrent").glob(
            ".persistent-process-group-*.json"
        )
    )


def test_persistent_slot_startup_timeout_terminates_live_child_and_cleans(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    component_parent = (tmp_path / "persistent_startup_timeout").resolve()
    tasks = _tasks(plan, component_parent, count=1)
    executor = PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=1,
        worker_target=f"{__name__}:_fake_spawn_owner_worker",
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
        poll_interval_seconds=0.0001,
        startup_timeout_seconds=0.0001,
    )

    with pytest.raises(
        RuntimeError,
        match="slot did not authenticate in time",
    ):
        executor.execute(
            tasks=tasks,
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=1,
        )

    assert not tuple(
        component_parent.glob(
            ".persistent-process-group-*.json"
        )
    )


def test_persistent_session_reuses_one_slot_across_executor_calls(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    tasks = _tasks(
        plan,
        (tmp_path / "persistent_session_components").resolve(),
    )
    marker_root = (tmp_path / "persistent_session_markers").resolve()
    executor = PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=1,
        startup_timeout_seconds=30.0,
        worker_target=f"{__name__}:_fake_spawn_owner_worker",
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
        poll_interval_seconds=0.01,
    )
    with executor.open_session(
        resources=("cpu",),
        max_workers=1,
        cpu_budget=2,
        marker_root=marker_root,
    ) as session:
        first = session.execute(
            tasks=(tasks[0],),
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=2,
        )[0]
        second = session.execute(
            tasks=(tasks[1],),
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=2,
        )[0]
        assert first.execution_telemetry["pid"] == (
            second.execution_telemetry["pid"]
        )
        assert first.execution_telemetry["slot_owner_ordinal"] == 1
        assert second.execution_telemetry["slot_owner_ordinal"] == 2
    assert not marker_root.exists()


def test_parent_reauthentication_rejects_child_tree_mutation(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=())
    task = _tasks(
        plan,
        (tmp_path / "tamper_components").resolve(),
        count=1,
    )[0]
    executor = ProcessIsolatedRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=1,
        worker_target=f"{__name__}:_fake_spawn_owner_worker",
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
        poll_interval_seconds=0.01,
    )
    result = executor.execute(
        tasks=(task,),
        worker=_must_not_execute_in_parent,
        max_workers=1,
        cpu_budget=1,
    )[0]
    _freshly_reauthenticate_owner_result(task=task, result=result)

    terminal = task.component_parent / "bow" / "execution_manifest.json"
    terminal.write_bytes(terminal.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source tree changed"):
        validate_authenticated_role_neutral_component_receipt(
            root=task.component_parent / "bow",
            plan=plan,
            physical_owner_scope_id=task.physical_owner.scope_id,
            receipt=result.sources[0].receipt,
            expected_component="bow",
        )
