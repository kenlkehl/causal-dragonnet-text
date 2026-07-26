from __future__ import annotations

import pytest

from oci.inference.stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
    ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    Stage1ExecutionTopologyPolicy,
)


def test_one_context_per_device_maps_each_device_and_counts_owner_slots() -> None:
    policy = Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_PER_SELECTED_DEVICE,
    )
    devices = ("cuda:4", "cuda:1", "cuda:9")

    assert {
        key: value.devices
        for key, value in policy.runtime_topologies(devices).items()
    } == {
        "cuda:4": ("cuda:4",),
        "cuda:1": ("cuda:1",),
        "cuda:9": ("cuda:9",),
    }
    assert (
        policy.effective_parallel_owners(
            devices=devices,
            workers_per_device=2,
        )
        == 6
    )
    assert policy.scientific_payload() == {
        "execution_topology_included_in_scientific_identity": False,
    }


def test_spanning_context_rotates_primary_and_does_not_multiply_owners() -> None:
    policy = Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    )
    devices = ("cuda:4", "cuda:1", "cuda:9")

    assert {
        key: value.devices
        for key, value in policy.runtime_topologies(devices).items()
    } == {
        "cuda:4": ("cuda:4", "cuda:1", "cuda:9"),
        "cuda:1": ("cuda:1", "cuda:4", "cuda:9"),
        "cuda:9": ("cuda:9", "cuda:4", "cuda:1"),
    }
    assert (
        policy.effective_parallel_owners(
            devices=devices,
            workers_per_device=2,
        )
        == 2
    )


@pytest.mark.parametrize(
    "devices",
    [
        ("cpu",),
        ("cuda:7",),
    ],
)
def test_spanning_context_rejects_non_multi_accelerator_selection(
    devices,
) -> None:
    policy = Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    )
    with pytest.raises(ValueError, match="at least two"):
        policy.runtime_topologies(devices)


def test_policy_mapping_is_closed_and_device_ids_are_not_defaults() -> None:
    value = {
        "schema_version": "portable_stage1_execution_topology_policy_v1",
        "mode": ONE_CONTEXT_PER_SELECTED_DEVICE,
    }
    assert Stage1ExecutionTopologyPolicy.from_mapping(value).as_dict() == value
    with pytest.raises(ValueError, match="every field"):
        Stage1ExecutionTopologyPolicy.from_mapping(
            {**value, "devices": ["cuda:0"]}
        )
    with pytest.raises(ValueError, match="unsupported"):
        Stage1ExecutionTopologyPolicy(mode="automatic_guess")


def test_cpu_one_context_policy_is_explicit_and_counted_without_gpu_assumptions() -> None:
    policy = Stage1ExecutionTopologyPolicy(
        mode=ONE_CONTEXT_PER_SELECTED_DEVICE,
    )
    assert policy.runtime_topologies(("cpu",))["cpu"].devices == ("cpu",)
    assert (
        policy.effective_parallel_owners(
            devices=("cpu",),
            workers_per_device=3,
        )
        == 3
    )
