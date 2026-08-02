"""Resource-portable single-node scheduling and execution attestations."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

from .portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
    Stage1ExecutionProfile,
    identity_sha256,
    normalize_device_policy,
)
from .stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
)

RESOURCE_INVENTORY_SCHEMA = "portable_single_node_resource_inventory_v1"
RESOURCE_PLAN_SCHEMA = "portable_single_node_resource_plan_v1"
EXECUTION_ATTESTATION_SCHEMA = "portable_execution_attestation_v1"
STAGE1_OWNER_CAPACITY_ATTESTATION_SCHEMA = (
    "production_stage1_owner_capacity_autodetection_v1"
)
GIB = 1024**3


def _numeric_cuda_visible_devices() -> tuple[int, ...] | None:
    """Return logical-to-physical CUDA indices for a numeric visibility mask."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    value = raw.strip()
    if not value or value == "-1":
        return ()
    parts = tuple(part.strip() for part in value.split(","))
    if any(not part or not part.isdigit() for part in parts):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must use unique numeric GPU indices; "
            "UUID and MIG masks are not supported by resource accounting"
        )
    indices = tuple(int(part) for part in parts)
    if len(indices) != len(set(indices)):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must use unique numeric GPU indices; "
            "UUID and MIG masks are not supported by resource accounting"
        )
    if os.environ.get("CUDA_DEVICE_ORDER", "").strip().upper() != "PCI_BUS_ID":
        raise ValueError(
            "numeric CUDA_VISIBLE_DEVICES resource accounting requires "
            "CUDA_DEVICE_ORDER=PCI_BUS_ID"
        )
    return indices


def _logical_to_physical_cuda_indices(
    logical_indices: Sequence[int],
) -> dict[int, int]:
    requested = tuple(int(index) for index in logical_indices)
    if any(index < 0 for index in requested):
        raise ValueError("logical CUDA indices must be nonnegative")
    visible = _numeric_cuda_visible_devices()
    if visible is None:
        return {index: index for index in requested}
    if any(index >= len(visible) for index in requested):
        raise RuntimeError(
            "requested logical CUDA device is excluded by "
            "CUDA_VISIBLE_DEVICES"
        )
    return {index: visible[index] for index in requested}


class Executor(Protocol):
    def submit(self, task: Any, *, resource: str) -> Any: ...


class ArtifactStore(Protocol):
    def publish(self, source: Any, *, artifact_id: str) -> Any: ...

    def resolve(self, artifact_id: str) -> Any: ...


@dataclass(frozen=True)
class GPUResource:
    device: str
    uuid: str
    total_memory_bytes: int
    free_memory_bytes: int
    utilization_percent: float
    external_processes: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.device.startswith("cuda:"):
            raise ValueError("GPU resource requires an explicit cuda:N device")
        if int(self.total_memory_bytes) < 1:
            raise ValueError("GPU total memory must be positive")
        if not 0 <= int(self.free_memory_bytes) <= int(self.total_memory_bytes):
            raise ValueError("GPU free memory is invalid")
        if not 0 <= float(self.utilization_percent) <= 100:
            raise ValueError("GPU utilization is invalid")

    @property
    def used_memory_bytes(self) -> int:
        return int(self.total_memory_bytes) - int(self.free_memory_bytes)


@dataclass(frozen=True)
class ResourceInventory:
    cpu_count: int
    gpus: tuple[GPUResource, ...]
    schema_version: str = RESOURCE_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RESOURCE_INVENTORY_SCHEMA:
            raise ValueError("unsupported resource inventory schema")
        if int(self.cpu_count) < 1:
            raise ValueError("resource inventory CPU count must be positive")
        devices = [gpu.device for gpu in self.gpus]
        if len(devices) != len(set(devices)):
            raise ValueError("resource inventory GPU devices are duplicated")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cpu_count": int(self.cpu_count),
            "gpus": [asdict(value) for value in self.gpus],
        }


def discover_resources() -> ResourceInventory:
    """Discover CPU/GPU resources without assuming a GPU count or fixed IDs."""

    cpu_count = max(1, int(os.cpu_count() or 1))
    visible = _numeric_cuda_visible_devices()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ResourceInventory(cpu_count=cpu_count, gpus=())
    process_map: dict[str, list[dict[str, Any]]] = {}
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        for line in processes.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                process_map.setdefault(parts[0], []).append(
                    {
                        "pid": int(parts[1]),
                        "used_memory_bytes": int(parts[2]) * 1024 * 1024,
                    }
                )
            except ValueError:
                continue
    except (OSError, subprocess.SubprocessError):
        pass
    physical_gpus: dict[int, GPUResource] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            physical_index = int(parts[0])
            physical_gpus[physical_index] = GPUResource(
                device=f"cuda:{physical_index}",
                uuid=parts[1],
                total_memory_bytes=int(parts[2]) * 1024 * 1024,
                free_memory_bytes=int(parts[3]) * 1024 * 1024,
                utilization_percent=float(parts[4]),
                external_processes=tuple(process_map.get(parts[1], ())),
            )
        except ValueError:
            continue
    if visible is None:
        gpus = tuple(physical_gpus.values())
    else:
        missing = [
            physical_index
            for physical_index in visible
            if physical_index not in physical_gpus
        ]
        if missing:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES references unavailable physical GPUs: "
                f"{missing}"
            )
        gpus = tuple(
            replace(
                physical_gpus[physical_index],
                device=f"cuda:{logical_index}",
            )
            for logical_index, physical_index in enumerate(visible)
        )
    return ResourceInventory(cpu_count=cpu_count, gpus=gpus)


def discover_host_available_memory_bytes() -> int:
    """Return currently available host memory without a platform constant."""

    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            rows = {
                fields[0].rstrip(":"): int(fields[1]) * 1024
                for line in stream
                if len(fields := line.split()) >= 2
                and fields[1].isdigit()
            }
        available = rows.get("MemAvailable")
        if isinstance(available, int) and available > 0:
            return available
    except OSError:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available = pages * page_size
    except (OSError, TypeError, ValueError):
        available = 0
    if available < 1:
        raise RuntimeError(
            "Stage 1 capacity autodetection cannot determine available "
            "host memory"
        )
    return available


@dataclass(frozen=True)
class ResourcePlan:
    devices: tuple[str, ...]
    cpu_budget: int
    inventory: ResourceInventory
    policy: tuple[str, ...]
    resource_performance_safety: ResourcePerformanceSafetyPolicy
    schema_version: str = RESOURCE_PLAN_SCHEMA

    def scientific_payload(self) -> Mapping[str, Any]:
        """Hardware is deliberately absent from scientific compatibility."""

        return {"runtime_compatibility": "single_node_executor_interface_v1"}

    def execution_attestation(self) -> Mapping[str, Any]:
        body = {
            "schema_version": EXECUTION_ATTESTATION_SCHEMA,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "selected_devices": list(self.devices),
            "cpu_budget": int(self.cpu_budget),
            "inventory": self.inventory.as_dict(),
            "resource_policy": list(self.policy),
            "resource_performance_safety": (self.resource_performance_safety.as_dict()),
        }
        return {**body, "content_sha256": identity_sha256(body)}


def plan_resources(
    *,
    policy: str | Sequence[str],
    cpu_budget: int,
    requested_device_count: int | None = None,
    inventory: ResourceInventory | None = None,
    cpu_supported: bool,
    resource_performance_safety: ResourcePerformanceSafetyPolicy,
) -> ResourcePlan:
    if not isinstance(
        resource_performance_safety,
        ResourcePerformanceSafetyPolicy,
    ):
        raise TypeError("resource planning requires typed resource_performance_safety")
    normalized = normalize_device_policy(policy)
    resources = inventory or discover_resources()
    selected: tuple[str, ...]
    if requested_device_count is not None and (
        isinstance(requested_device_count, bool)
        or not isinstance(requested_device_count, int)
        or requested_device_count < 1
    ):
        raise ValueError("requested_device_count must be a positive integer")
    if int(cpu_budget) < 1 or int(cpu_budget) > int(resources.cpu_count):
        raise ValueError(
            f"CPU budget {cpu_budget} exceeds available host CPUs {resources.cpu_count}"
        )
    by_device = {gpu.device: gpu for gpu in resources.gpus}
    if normalized == ("cpu",):
        if not cpu_supported:
            raise RuntimeError("selected scientific models do not permit CPU execution")
        if requested_device_count not in (None, 1):
            raise ValueError("CPU resource policy supports exactly one execution device")
        selected = ("cpu",)
    else:
        requested = tuple(by_device) if normalized == ("auto",) else tuple(normalized)
        if (
            normalized != ("auto",)
            and requested_device_count is not None
            and requested_device_count != len(requested)
        ):
            raise ValueError(
                "requested_device_count must equal the number of explicitly "
                "configured accelerator devices"
            )
        missing = [value for value in requested if value not in by_device]
        if missing:
            if (
                normalized == ("auto",)
                and cpu_supported
                and not by_device
                and requested_device_count in (None, 1)
            ):
                selected = ("cpu",)
            else:
                raise RuntimeError(
                    "requested accelerator inventory is unavailable: "
                    f"missing={missing}, available={sorted(by_device)}"
                )
        else:
            acceptable: list[str] = []
            rejected: list[dict[str, Any]] = []
            for device in requested:
                gpu = by_device[device]
                reasons = []
                if (
                    resource_performance_safety.fail_on_external_gpu_occupants
                    and gpu.external_processes
                ):
                    reasons.append("external_compute_occupant")
                if int(gpu.free_memory_bytes) < int(
                    resource_performance_safety.gpu_minimum_headroom_bytes
                ):
                    reasons.append("less_than_required_headroom")
                if (
                    gpu.used_memory_bytes / gpu.total_memory_bytes
                    > resource_performance_safety.gpu_max_allocation_fraction
                ):
                    reasons.append("existing_allocation_exceeds_fraction")
                if reasons:
                    rejected.append(
                        {
                            "device": device,
                            "reasons": reasons,
                            "resource": asdict(gpu),
                        }
                    )
                else:
                    acceptable.append(device)
            if rejected and normalized != ("auto",):
                raise RuntimeError(
                    "requested resources are occupied or unsafe; no external process "
                    f"was killed. report={rejected}"
                )
            if acceptable:
                if (
                    requested_device_count is not None
                    and len(acceptable) < requested_device_count
                ):
                    raise RuntimeError(
                        "the auto resource policy found fewer safe accelerators "
                        "than the deployment-selected device count; no external "
                        f"process was killed. requested={requested_device_count}, "
                        f"available={len(acceptable)}, rejected={rejected}"
                    )
                selected_count = requested_device_count or len(acceptable)
                selected = tuple(acceptable[:selected_count])
            elif (
                normalized == ("auto",)
                and cpu_supported
                and requested_device_count in (None, 1)
            ):
                selected = ("cpu",)
            else:
                raise RuntimeError(
                    "no compatible resources satisfy the allocation policy; no "
                    f"external process was killed. report={rejected}"
                )
    return ResourcePlan(
        devices=selected,
        cpu_budget=int(cpu_budget),
        inventory=resources,
        policy=normalized,
        resource_performance_safety=resource_performance_safety,
    )


def resolve_stage1_owner_capacity(
    *,
    profile: Stage1ExecutionProfile,
    resource_plan: ResourcePlan,
    host_available_memory_bytes: int | None = None,
) -> tuple[Stage1ExecutionProfile, Mapping[str, Any]]:
    """Resolve uniform persistent owner lanes from live resource capacity.

    The configured profile always remains the upper bound.  Uniform lanes are
    intentional: the persistent executor currently provisions one equal lane
    count on every selected device, so selecting the minimum safe GPU capacity
    prevents a smaller device from being overcommitted.
    """

    if not isinstance(profile, Stage1ExecutionProfile):
        raise TypeError(
            "Stage 1 owner-capacity resolution requires a typed profile"
        )
    if not isinstance(resource_plan, ResourcePlan):
        raise TypeError(
            "Stage 1 owner-capacity resolution requires a resource plan"
        )
    selected = tuple(resource_plan.devices)
    if len(selected) != int(profile.device_count):
        raise ValueError(
            "Stage 1 owner-capacity inventory differs from the configured "
            "device count"
        )
    capacity_policy = profile.owner_capacity_policy
    configured = {
        "scope_workers_per_device_ceiling": int(
            profile.scope_workers_per_device
        ),
        "max_parallel_owners_ceiling": int(
            profile.max_parallel_owners
        ),
    }
    if capacity_policy.mode == "fixed":
        effective = profile
        body = {
            "schema_version": (
                STAGE1_OWNER_CAPACITY_ATTESTATION_SCHEMA
            ),
            "mode": "fixed",
            "selected_devices": list(selected),
            "configured_capacity": configured,
            "owner_capacity_policy": capacity_policy.as_dict(),
            "host_available_memory_bytes": None,
            "host_owner_lane_cap": None,
            "cpu_owner_lane_cap": int(resource_plan.cpu_budget),
            "per_device_capacity": [],
            "effective_scope_workers_per_device": int(
                effective.scope_workers_per_device
            ),
            "effective_max_parallel_owners": int(
                effective.max_parallel_owners
            ),
            "configured_values_are_hard_ceilings": True,
            "resource_assignment_in_scientific_identity": False,
            "completion_order_in_scientific_identity": False,
        }
        return effective, {
            **body,
            "content_sha256": identity_sha256(body),
        }

    available_host = (
        discover_host_available_memory_bytes()
        if host_available_memory_bytes is None
        else int(host_available_memory_bytes)
    )
    if available_host < 1:
        raise ValueError(
            "Stage 1 owner-capacity host memory must be positive"
        )
    host_budget = int(
        available_host
        * capacity_policy.host_memory_budget_fraction
    )
    host_owner_cap = (
        host_budget
        // capacity_policy.estimated_host_memory_bytes_per_owner
    )
    cpu_owner_cap = (
        int(resource_plan.cpu_budget)
        // capacity_policy.minimum_cpu_threads_per_owner
    )
    if host_owner_cap < 1 or cpu_owner_cap < 1:
        raise RuntimeError(
            "Stage 1 capacity autodetection leaves no host-memory or CPU "
            "owner lane"
        )

    per_device_rows: list[dict[str, Any]] = []
    if profile.resource_kind == "accelerator":
        inventory_by_device = {
            gpu.device: gpu for gpu in resource_plan.inventory.gpus
        }
        for device in selected:
            gpu = inventory_by_device.get(device)
            if gpu is None:
                raise RuntimeError(
                    "Stage 1 capacity autodetection lacks a selected GPU "
                    f"inventory row: {device}"
                )
            free_after_reserve = max(
                0,
                int(gpu.free_memory_bytes)
                - capacity_policy.device_memory_reserve_bytes,
            )
            # Resource admission already applied the deployment's existing
            # allocation threshold.  Lane sizing is based on the memory that
            # is actually free after the separately configured reserve; using
            # the admission fraction again would incorrectly turn a
            # "GPU must be 90% free" rule into a 10%-of-VRAM workflow cap.
            allocatable = free_after_reserve
            detected_lanes = (
                allocatable
                // capacity_policy
                .estimated_device_memory_bytes_per_owner
            )
            per_device_rows.append(
                {
                    "device": device,
                    "uuid": gpu.uuid,
                    "total_memory_bytes": int(
                        gpu.total_memory_bytes
                    ),
                    "free_memory_bytes": int(gpu.free_memory_bytes),
                    "used_memory_bytes": int(gpu.used_memory_bytes),
                    "free_after_reserve_bytes": free_after_reserve,
                    "allocatable_owner_memory_bytes": allocatable,
                    "detected_owner_lane_cap": int(detected_lanes),
                }
            )
        device_lane_cap = min(
            int(row["detected_owner_lane_cap"])
            for row in per_device_rows
        )
    elif selected == ("cpu",):
        device_lane_cap = int(profile.scope_workers_per_device)
    else:
        raise ValueError(
            "Stage 1 capacity autodetection resource kind conflicts with "
            "the selected devices"
        )
    if device_lane_cap < 1:
        raise RuntimeError(
            "Stage 1 capacity autodetection found no safe per-device owner "
            "lane"
        )

    if (
        profile.neural_query_topology.mode
        == ONE_CONTEXT_PER_SELECTED_DEVICE
    ):
        device_count = len(selected)
        uniform_workers = min(
            int(profile.scope_workers_per_device),
            int(device_lane_cap),
            int(host_owner_cap) // device_count,
            int(cpu_owner_cap) // device_count,
            int(profile.max_parallel_owners) // device_count,
        )
        if uniform_workers < 1:
            raise RuntimeError(
                "Stage 1 capacity autodetection cannot provision one "
                "uniform lane on every selected device; select fewer "
                "devices or increase the host budget"
            )
        topology_owner_cap = device_count * uniform_workers
    else:
        uniform_workers = min(
            int(profile.scope_workers_per_device),
            int(device_lane_cap),
            int(host_owner_cap),
            int(cpu_owner_cap),
            int(profile.max_parallel_owners),
        )
        topology_owner_cap = uniform_workers
    effective_owner_cap = min(
        int(profile.max_parallel_owners),
        int(host_owner_cap),
        int(cpu_owner_cap),
        int(topology_owner_cap),
    )
    if effective_owner_cap < 1:
        raise RuntimeError(
            "Stage 1 capacity autodetection leaves no executable owner"
        )
    effective_preflight = replace(
        profile.preflight_execution_policy,
        max_parallel_owners=min(
            int(
                profile.preflight_execution_policy
                .max_parallel_owners
            ),
            effective_owner_cap,
        ),
    )
    effective = replace(
        profile,
        scope_workers_per_device=uniform_workers,
        max_parallel_owners=effective_owner_cap,
        preflight_execution_policy=effective_preflight,
    )
    body = {
        "schema_version": STAGE1_OWNER_CAPACITY_ATTESTATION_SCHEMA,
        "mode": "resource_autodetect",
        "selected_devices": list(selected),
        "configured_capacity": configured,
        "owner_capacity_policy": capacity_policy.as_dict(),
        "host_available_memory_bytes": available_host,
        "host_memory_budget_bytes": host_budget,
        "host_owner_lane_cap": int(host_owner_cap),
        "cpu_owner_lane_cap": int(cpu_owner_cap),
        "per_device_capacity": per_device_rows,
        "minimum_uniform_device_lane_cap": int(device_lane_cap),
        "topology_owner_lane_cap": int(topology_owner_cap),
        "effective_scope_workers_per_device": int(
            effective.scope_workers_per_device
        ),
        "effective_max_parallel_owners": int(
            effective.max_parallel_owners
        ),
        "effective_preflight_max_parallel_owners": int(
            effective.preflight_execution_policy.max_parallel_owners
        ),
        "configured_values_are_hard_ceilings": True,
        "resource_assignment_in_scientific_identity": False,
        "completion_order_in_scientific_identity": False,
    }
    return effective, {
        **body,
        "content_sha256": identity_sha256(body),
    }


def assign_physical_fits(
    physical_fit_keys: Sequence[str],
    plan: ResourcePlan,
) -> Mapping[str, str]:
    """Deterministic device assignment independent of completion order."""

    keys = tuple(str(value) for value in physical_fit_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("physical-fit scheduler keys are duplicated")
    if not plan.devices:
        raise ValueError("resource plan has no execution devices")
    return {key: plan.devices[index % len(plan.devices)] for index, key in enumerate(sorted(keys))}


@dataclass(frozen=True)
class BenchmarkCandidate:
    name: str
    device_count: int
    concurrency_per_device: int
    throughput_scopes_per_second: float
    single_device_baseline_throughput: float
    peak_allocation_fraction: float
    minimum_headroom_bytes: int
    repeated_runs: int
    oom_count: int
    deterministic: bool
    scientifically_equal: bool


def select_fastest_safe_candidate(
    candidates: Sequence[BenchmarkCandidate],
    *,
    resource_performance_safety: ResourcePerformanceSafetyPolicy,
) -> BenchmarkCandidate:
    """Select from measured candidates using explicit deployment gates."""

    if not isinstance(
        resource_performance_safety,
        ResourcePerformanceSafetyPolicy,
    ):
        raise TypeError("benchmark selection requires typed resource_performance_safety")
    accepted: list[BenchmarkCandidate] = []
    for value in candidates:
        speedup = (
            float(value.throughput_scopes_per_second)
            / float(value.single_device_baseline_throughput)
            if value.single_device_baseline_throughput > 0
            else 0.0
        )
        if (
            value.oom_count == 0
            and int(value.repeated_runs)
            >= resource_performance_safety.minimum_benchmark_repetitions_per_scope
            and value.deterministic
            and value.scientifically_equal
            and float(value.peak_allocation_fraction)
            <= resource_performance_safety.gpu_max_allocation_fraction
            and int(value.minimum_headroom_bytes)
            >= resource_performance_safety.gpu_minimum_headroom_bytes
            and (
                int(value.device_count) <= 1
                or speedup >= resource_performance_safety.minimum_multi_device_throughput_ratio
            )
        ):
            accepted.append(value)
    if not accepted:
        raise RuntimeError("no benchmark configuration satisfies resource/equality gates")
    # Exact throughput ties select the lower total concurrency.
    return min(
        accepted,
        key=lambda value: (
            -float(value.throughput_scopes_per_second),
            int(value.device_count) * int(value.concurrency_per_device),
            int(value.device_count),
            value.name,
        ),
    )


__all__ = [
    "ArtifactStore",
    "BenchmarkCandidate",
    "EXECUTION_ATTESTATION_SCHEMA",
    "Executor",
    "GIB",
    "GPUResource",
    "RESOURCE_INVENTORY_SCHEMA",
    "RESOURCE_PLAN_SCHEMA",
    "STAGE1_OWNER_CAPACITY_ATTESTATION_SCHEMA",
    "ResourceInventory",
    "ResourcePlan",
    "assign_physical_fits",
    "discover_resources",
    "discover_host_available_memory_bytes",
    "plan_resources",
    "resolve_stage1_owner_capacity",
    "select_fastest_safe_candidate",
]
