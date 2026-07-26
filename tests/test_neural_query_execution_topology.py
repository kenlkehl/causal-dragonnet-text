from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oci.inference.neural_query_execution_topology import (
    NeuralQueryExecutionTopology,
)
from oci.inference.production_role_neutral_persistent_executor import (
    PersistentSpawnRoleNeutralPhysicalOwnerExecutor,
    PersistentRoleNeutralExecutionSession,
)
from oci.inference.production_role_neutral_process_executor import (
    _change_resource_tuple_reservation,
    _resource_tuple_has_capacity,
    _runtime_neural_query_topology_attestation,
)
from oci.inference.production_stage1_role_neutral_execution import (
    RoleNeutralComponentInvocation,
    RoleNeutralStage1ExecutionPolicy,
)
from tests.test_production_stage1_role_neutral_execution import (
    _plan,
    _resource_plan,
)
from tests.test_production_role_neutral_process_executor import (
    _must_not_execute_in_parent,
    _tasks,
)


def test_topology_rejects_duplicate_mixed_and_implicit_devices() -> None:
    with pytest.raises(ValueError, match="cannot be duplicated"):
        NeuralQueryExecutionTopology(
            devices=("cuda:2", "cuda:2")
        )
    with pytest.raises(ValueError, match="CPU cannot be mixed"):
        NeuralQueryExecutionTopology(
            devices=("cpu", "cuda:2")
        )
    with pytest.raises(ValueError, match="explicit cpu/cuda"):
        NeuralQueryExecutionTopology(devices=("gpu",))


def test_deployment_policy_rejects_unselected_topology_resources() -> None:
    resources = _resource_plan(
        devices=("cuda:2",),
        cpu_budget=2,
    )
    with pytest.raises(ValueError, match="unavailable resource"):
        RoleNeutralStage1ExecutionPolicy(
            resource_plan=resources,
            max_parallel_owners=1,
            neural_query_execution_topologies={
                "cuda:2": NeuralQueryExecutionTopology(
                    devices=("cuda:2", "cuda:7")
                )
            },
        )


def test_invocation_scientific_identity_excludes_device_topology(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=(2, 7))
    owner, members = plan.physical_scope_groups[0]
    common = {
        "plan": plan,
        "physical_owner": owner,
        "logical_members": members,
        "component": "neural_query",
        "resource": "cuda:2",
    }
    single = RoleNeutralComponentInvocation(
        **common,
        output_root=(tmp_path / "single").resolve(),
        neural_query_execution_topology=(
            NeuralQueryExecutionTopology.single("cuda:2")
        ),
    )
    spanned = RoleNeutralComponentInvocation(
        **common,
        output_root=(tmp_path / "spanned").resolve(),
        neural_query_execution_topology=(
            NeuralQueryExecutionTopology(
                devices=("cuda:2", "cuda:7")
            )
        ),
    )

    assert single.scientific_payload() == spanned.scientific_payload()
    assert (
        single.scientific_payload()[
            "neural_query_device_topology_included"
        ]
        is False
    )


class _FakeCuda:
    def __init__(self, properties):
        self._properties = tuple(properties)

    @staticmethod
    def is_available() -> bool:
        return True

    def device_count(self) -> int:
        return len(self._properties)

    def get_device_properties(self, index: int):
        return self._properties[index]


def _properties(*, name: str, memory: int = 24_000):
    return SimpleNamespace(
        name=name,
        major=8,
        minor=6,
        total_memory=memory,
        multi_processor_count=84,
    )


def test_runtime_topology_fails_closed_on_unavailable_and_heterogeneous_gpus() -> None:
    topology = NeuralQueryExecutionTopology(
        devices=("cuda:0", "cuda:1")
    )
    unavailable = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_properties=lambda _index: None,
        )
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        _runtime_neural_query_topology_attestation(
            topology,
            torch_module=unavailable,
        )

    heterogeneous = SimpleNamespace(
        cuda=_FakeCuda(
            (
                _properties(name="first"),
                _properties(name="second"),
            )
        )
    )
    with pytest.raises(RuntimeError, match="heterogeneous"):
        _runtime_neural_query_topology_attestation(
            topology,
            torch_module=heterogeneous,
        )


def test_runtime_topology_attests_homogeneous_tuple_without_scientific_binding() -> None:
    topology = NeuralQueryExecutionTopology(
        devices=("cuda:4", "cuda:1")
    )
    properties = tuple(_properties(name="same") for _ in range(5))
    attestation = _runtime_neural_query_topology_attestation(
        topology,
        torch_module=SimpleNamespace(cuda=_FakeCuda(properties)),
    )
    assert attestation["devices"] == ["cuda:4", "cuda:1"]
    assert attestation["homogeneous"] is True
    assert attestation["scientific_identity_includes_topology"] is False


def test_persistent_session_reserves_every_spanned_device_atomically() -> None:
    session = object.__new__(PersistentRoleNeutralExecutionSession)
    session._condition = threading.Condition()
    session._closed = False
    session._broken = None
    session._active_calls = 0
    slots = [
        SimpleNamespace(resource="cuda:3", busy=False),
        SimpleNamespace(resource="cuda:9", busy=False),
    ]
    session._slots = slots

    reserved = session._acquire(("cuda:3", "cuda:9"))
    assert tuple(slot.resource for slot in reserved) == (
        "cuda:3",
        "cuda:9",
    )
    assert all(slot.busy for slot in slots)
    assert session._active_calls == 1

    session._release(reserved, failure=None)
    assert not any(slot.busy for slot in slots)
    assert session._active_calls == 0


def test_fresh_executor_reservation_ledger_blocks_every_spanned_device() -> None:
    active: dict[str, int] = {}
    spanned = ("cuda:5", "cuda:2")
    assert _resource_tuple_has_capacity(
        spanned,
        active_by_resource=active,
        maximum_per_resource=1,
    )
    _change_resource_tuple_reservation(
        spanned,
        active_by_resource=active,
        delta=1,
    )
    assert active == {"cuda:5": 1, "cuda:2": 1}
    assert not _resource_tuple_has_capacity(
        ("cuda:2",),
        active_by_resource=active,
        maximum_per_resource=1,
    )
    assert not _resource_tuple_has_capacity(
        ("cuda:5",),
        active_by_resource=active,
        maximum_per_resource=1,
    )
    _change_resource_tuple_reservation(
        spanned,
        active_by_resource=active,
        delta=-1,
    )
    assert active == {"cuda:5": 0, "cuda:2": 0}


def test_non_session_persistent_shortcut_rejects_device_span(
    tmp_path: Path,
) -> None:
    plan = _plan(gpu_ids=(0, 1))
    task = _tasks(
        plan,
        (tmp_path / "components").resolve(),
        count=1,
    )[0]
    task = replace(
        task,
        resource="cuda:0",
        neural_query_execution_topology=(
            NeuralQueryExecutionTopology(
                devices=("cuda:0", "cuda:1")
            )
        ),
    )
    executor = PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
        max_workers_per_resource=1,
        startup_timeout_seconds=30.0,
        worker_target=(
            "tests.test_production_role_neutral_process_executor:"
            "_fake_spawn_owner_worker"
        ),
        worker_parameters={"test_token": "mixed-seed-isolation"},
        production_worker_required=False,
    )

    with pytest.raises(RuntimeError, match="reserve.*atomically"):
        executor.execute(
            tasks=(task,),
            worker=_must_not_execute_in_parent,
            max_workers=1,
            cpu_budget=1,
        )
