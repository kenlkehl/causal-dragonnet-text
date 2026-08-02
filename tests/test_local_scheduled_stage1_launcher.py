from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from oci.inference.portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
    plan_resources,
)


ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = ROOT / "scripts/build_local_stage1_deployment_profile.py"
    spec = importlib.util.spec_from_file_location(
        "test_build_local_stage1_deployment_profile",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_parallel_launcher_enables_production_capacity_autodetection(
) -> None:
    launcher = (
        ROOT / "run_five_conf_five_mod_local_parallel.sh"
    ).read_text(encoding="utf-8")

    assert 'scope_workers_per_device="${SCOPE_WORKERS_PER_DEVICE:-4}"' in launcher
    assert "--owner-capacity-mode resource_autodetect" in launcher
    for option in (
        "--estimated-device-memory-per-owner",
        "--device-memory-reserve",
        "--estimated-host-memory-per-owner",
        "--host-memory-budget-fraction",
        "--minimum-cpu-threads-per-owner",
    ):
        assert option in launcher
    assert "stage1_owner_capacity_attestation" in launcher


def _args(
    *,
    tmp_path: Path,
    target: Path,
    durable: Path,
    devices: tuple[str, ...],
) -> argparse.Namespace:
    parser = _builder().build_parser()
    values = [
        "--base",
        str(
            ROOT
            / "example_configs/portable_all_evidence_deployment_nsclc.stage1-only.example.json"
        ),
        "--target",
        str(target),
        "--dataset",
        str(tmp_path / "dataset.parquet"),
        "--durable-root",
        str(durable),
        "--scratch-root",
        str(tmp_path / "shared-scratch"),
        "--embedding-model",
        str(tmp_path / "embedding-model"),
        "--htr-model",
        str(tmp_path / "htr-model"),
        "--stage1-profile",
        str(ROOT / "example_configs/production_all_evidence_stage1_full.json"),
        "--query-profile",
        str(
            ROOT
            / "example_configs/production_all_evidence_neural_query_full.json"
        ),
        "--cpu-budget",
        "8",
        "--preflight-memory-budget",
        str(32 * 1024**3),
        "--preflight-owner-peak",
        str(8 * 1024**3),
        "--preflight-lanes",
        str(len(devices)),
        "--embedding-batch-size",
        "8",
        "--gpu-minimum-free-fraction",
        "0.90",
    ]
    for device in devices:
        values.extend(("--device", device))
    return parser.parse_args(values)


def test_local_profiles_compile_four_then_two_disjoint_lanes(
    tmp_path: Path,
) -> None:
    module = _builder()
    shared = tmp_path / "shared-scratch"
    before = module.build_profile(
        _args(
            tmp_path=tmp_path,
            target=tmp_path / "profiles/gpu0123.json",
            durable=tmp_path / "runs/gpu0123",
            devices=("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        )
    )
    after = module.build_profile(
        _args(
            tmp_path=tmp_path,
            target=tmp_path / "profiles/gpu23.json",
            durable=tmp_path / "runs/gpu23",
            devices=("cuda:0", "cuda:1"),
        )
    )

    assert tuple(before.devices) == (
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
    )
    assert before.stage1_execution.max_parallel_owners == 4
    assert (
        before.stage1_execution.owner_capacity_policy.mode
        == "resource_autodetect"
    )
    assert tuple(after.devices) == ("cuda:0", "cuda:1")
    assert after.stage1_execution.max_parallel_owners == 2
    assert before.scratch_root == shared
    assert after.scratch_root == shared
    assert before.durable_artifact_root != after.durable_artifact_root
    assert before.endpoint is None and after.endpoint is None
    assert before.oracle_source is None and after.oracle_source is None
    assert (
        before.resource_performance_safety.fail_on_external_gpu_occupants
        is False
    )
    assert before.resource_performance_safety.gpu_max_allocation_fraction == 0.1
    assert before.resource_performance_safety.gpu_minimum_headroom_bytes == 0


def test_local_profile_rejects_owner_cap_above_resource_capacity(
    tmp_path: Path,
) -> None:
    module = _builder()
    args = _args(
        tmp_path=tmp_path,
        target=tmp_path / "profile.json",
        durable=tmp_path / "run",
        devices=("cuda:0", "cuda:1"),
    )
    args.max_parallel_owners = 3

    with pytest.raises(ValueError, match="device or CPU capacity"):
        module.build_profile(args)


def test_local_policy_allows_occupant_at_ninety_percent_free(
    tmp_path: Path,
) -> None:
    profile = _builder().build_profile(
        _args(
            tmp_path=tmp_path,
            target=tmp_path / "profile.json",
            durable=tmp_path / "run",
            devices=("cuda:0",),
        )
    )
    accepted_inventory = ResourceInventory(
        cpu_count=8,
        gpus=(
            GPUResource(
                device="cuda:0",
                uuid="gpu-0",
                total_memory_bytes=1_000,
                free_memory_bytes=900,
                utilization_percent=4.0,
                external_processes=(
                    {"pid": 123, "used_memory_bytes": 100},
                ),
            ),
        ),
    )

    accepted = plan_resources(
        policy=("cuda:0",),
        cpu_budget=8,
        requested_device_count=1,
        inventory=accepted_inventory,
        cpu_supported=False,
        resource_performance_safety=(
            profile.resource_performance_safety
        ),
    )
    assert accepted.devices == ("cuda:0",)

    rejected_inventory = ResourceInventory(
        cpu_count=8,
        gpus=(
            GPUResource(
                device="cuda:0",
                uuid="gpu-0",
                total_memory_bytes=1_000,
                free_memory_bytes=899,
                utilization_percent=4.0,
                external_processes=(
                    {"pid": 123, "used_memory_bytes": 101},
                ),
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="occupied or unsafe"):
        plan_resources(
            policy=("cuda:0",),
            cpu_budget=8,
            requested_device_count=1,
            inventory=rejected_inventory,
            cpu_supported=False,
            resource_performance_safety=(
                profile.resource_performance_safety
            ),
        )
