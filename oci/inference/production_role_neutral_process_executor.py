"""Spawn-isolated execution for role-neutral Stage 1 physical owners.

The role-neutral producers reset NumPy, Torch, and Python RNG state at a
physical-owner boundary.  They therefore cannot safely share one interpreter
when owners with different seeds run concurrently.  This module provides the
production single-node executor used by the portable workflow:

* every owner runs in a fresh ``spawn`` child with its own Python hash seed;
* the child rehydrates one sealed ``PreparedStage1Context`` without rerunning
  monolithic preparation;
* native thread pools share the caller's explicit host CPU budget;
* one child is bound to one explicit CPU/CUDA resource;
* the child executes and authenticates all six role-neutral components; and
* only typed receipts and operational telemetry cross the process boundary.

The parent coordinator remains responsible for freshly reopening every
returned component tree.  No pickle is persisted as an artifact; Python's
standard multiprocessing transport is used only for ephemeral local IPC.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import re
import resource
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .production_stage1_role_neutral_execution import (
    RoleNeutralPhysicalOwnerResult,
    RoleNeutralPhysicalOwnerTask,
    _execute_one_owner,
)
from .neural_query_execution_topology import (
    NeuralQueryExecutionTopology,
)
from .production_stage1_scope_scheduler import (
    _enforce_stage1_torch_determinism,
    _establish_worker_process_group,
    _observe_stage1_torch_determinism,
    _terminate_process_and_descendants,
    seed_stage1_scope_rngs,
)
from .prepared_stage1_context import (
    _option_mapping,
    _options_from_mapping,
)


PROCESS_ISOLATED_ROLE_NEUTRAL_AUTHORITY_SCHEMA = (
    "production_role_neutral_process_worker_authority_v1"
)
PROCESS_ISOLATED_ROLE_NEUTRAL_EXECUTOR_SCHEMA = (
    "production_role_neutral_process_executor_v1"
)
PRODUCTION_PROCESS_WORKER_TARGET = (
    "oci.inference.production_role_neutral_process_executor:"
    "_execute_production_role_neutral_owner"
)

_TARGET = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_HEX = frozenset("0123456789abcdef")
_PROCESS_START_ENVIRONMENT_LOCK = threading.Lock()
_NATIVE_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        payload = _canonical_json(value)
        return json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be closed finite JSON") from exc


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


@dataclass(frozen=True)
class PreparedRoleNeutralProcessAuthority:
    """Closed capability for reconstructing one authenticated prepared build."""

    stage1_build_options: Mapping[str, Any]
    architecture_profiles: Mapping[str, Mapping[str, Any]]
    runtime_compatibility_class: str
    expected_prepared_request_sha256: str
    expected_full_plan: Mapping[str, Any]
    schema_version: str = PROCESS_ISOLATED_ROLE_NEUTRAL_AUTHORITY_SCHEMA
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROCESS_ISOLATED_ROLE_NEUTRAL_AUTHORITY_SCHEMA:
            raise ValueError("unsupported process-worker authority schema")
        options = _json_copy(
            self.stage1_build_options,
            label="stage1_build_options",
        )
        profiles = _json_copy(
            self.architecture_profiles,
            label="architecture_profiles",
        )
        plan = _json_copy(self.expected_full_plan, label="expected_full_plan")
        _options_from_mapping(options)
        request_sha256 = _require_sha256(
            self.expected_prepared_request_sha256,
            label="expected prepared request",
        )
        runtime = str(self.runtime_compatibility_class).strip()
        if not runtime:
            raise ValueError("runtime compatibility class must be nonempty")
        body = {
            "schema_version": self.schema_version,
            "stage1_build_options": options,
            "architecture_profiles": profiles,
            "runtime_compatibility_class": runtime,
            "expected_prepared_request_sha256": request_sha256,
            "expected_full_plan": plan,
        }
        object.__setattr__(self, "stage1_build_options", options)
        object.__setattr__(self, "architecture_profiles", profiles)
        object.__setattr__(self, "runtime_compatibility_class", runtime)
        object.__setattr__(
            self,
            "expected_prepared_request_sha256",
            request_sha256,
        )
        object.__setattr__(self, "expected_full_plan", plan)
        object.__setattr__(self, "content_sha256", _sha256_json(body))

    @classmethod
    def from_prepared(
        cls,
        *,
        prepared: Any,
        producer_factories_builder: Any,
    ) -> "PreparedRoleNeutralProcessAuthority":
        from .production_role_neutral_producer_factories import (
            PreparedBuildRoleNeutralProducerFactoriesBuilder,
        )
        from .production_stage1_bundle import _PreparedBuild

        if not isinstance(prepared, _PreparedBuild):
            raise TypeError("process authority requires one typed prepared build")
        if not isinstance(
            producer_factories_builder,
            PreparedBuildRoleNeutralProducerFactoriesBuilder,
        ):
            raise TypeError(
                "production process authority requires the exact prepared-build "
                "producer factory builder"
            )
        return cls(
            stage1_build_options=_option_mapping(prepared),
            architecture_profiles=producer_factories_builder.architecture_profiles,
            runtime_compatibility_class=(
                producer_factories_builder.runtime_compatibility_class
            ),
            expected_prepared_request_sha256=prepared.request_sha256,
            expected_full_plan=prepared.stage1_scope_plan.as_dict(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage1_build_options": copy.deepcopy(
                dict(self.stage1_build_options)
            ),
            "architecture_profiles": copy.deepcopy(
                dict(self.architecture_profiles)
            ),
            "runtime_compatibility_class": self.runtime_compatibility_class,
            "expected_prepared_request_sha256": (
                self.expected_prepared_request_sha256
            ),
            "expected_full_plan": copy.deepcopy(dict(self.expected_full_plan)),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PreparedRoleNeutralProcessAuthority":
        required = {
            "schema_version",
            "stage1_build_options",
            "architecture_profiles",
            "runtime_compatibility_class",
            "expected_prepared_request_sha256",
            "expected_full_plan",
            "content_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("process-worker authority is not one closed object")
        authority = cls(
            schema_version=str(value["schema_version"]),
            stage1_build_options=value["stage1_build_options"],
            architecture_profiles=value["architecture_profiles"],
            runtime_compatibility_class=str(
                value["runtime_compatibility_class"]
            ),
            expected_prepared_request_sha256=str(
                value["expected_prepared_request_sha256"]
            ),
            expected_full_plan=value["expected_full_plan"],
        )
        if authority.content_sha256 != _require_sha256(
            value["content_sha256"],
            label="process-worker authority content",
        ):
            raise ValueError("process-worker authority content hash changed")
        return authority

    def reconstruct(self) -> tuple[Any, Any]:
        """Reject the retired monolithic per-owner reconstruction path."""

        raise RuntimeError(
            "legacy process authority cannot reconstruct Stage 1; bind one "
            "sealed PreparedStage1Context artifact"
        )


def _plan_is_exact_projection(full_plan: Any, projected: Any) -> bool:
    """Prove a one/multi-owner task plan is an exact group projection."""

    if (
        full_plan.registry_content_sha256
        != projected.registry_content_sha256
        or full_plan.global_seed != projected.global_seed
        or full_plan.review_rounds != projected.review_rounds
        or full_plan.initial_training_partitions
        != projected.initial_training_partitions
        or full_plan.physical_fit_identity
        != projected.physical_fit_identity
    ):
        return False
    full_scopes = {
        scope.scope_id: scope.as_dict() for scope in full_plan.scopes
    }
    projected_ids = {scope.scope_id for scope in projected.scopes}
    if not projected_ids or any(
        full_scopes.get(scope.scope_id) != scope.as_dict()
        for scope in projected.scopes
    ):
        return False
    for owner, members in projected.physical_scope_groups:
        full_owner = full_plan.physical_owner(owner.scope_id)
        full_group = next(
            group
            for candidate, group in full_plan.physical_scope_groups
            if candidate.scope_id == full_owner.scope_id
        )
        if (
            owner != full_owner
            or tuple(member.scope_id for member in members)
            != tuple(member.scope_id for member in full_group)
            or not {member.scope_id for member in full_group}.issubset(
                projected_ids
            )
        ):
            return False
    return True


def _execute_production_role_neutral_owner(
    *,
    task: RoleNeutralPhysicalOwnerTask,
    worker_parameters: Mapping[str, Any],
) -> RoleNeutralPhysicalOwnerResult:
    """Importable production worker target executed only inside a spawn child."""

    parameter_fields = set(worker_parameters)
    if parameter_fields == {"prepared_context_manifest_path"}:
        from .prepared_stage1_context import load_prepared_stage1_context

        context = load_prepared_stage1_context(
            Path(str(worker_parameters["prepared_context_manifest_path"]))
        )
        # A fresh-per-fit child owns the full host budget assigned to that fit.
        # Native pools are already bounded by the child entry.
        native_budget = int(
            os.environ.get("OMP_NUM_THREADS", "1")
        )
        prepared, factories = context.reconstruct(
            slot_cpu_budget=native_budget,
        )
    else:
        raise ValueError("production owner worker parameters are not closed")
    if not _plan_is_exact_projection(
        prepared.stage1_scope_plan,
        task.plan,
    ):
        raise ValueError(
            "spawned owner task plan is not an exact prepared-plan projection"
        )
    return _execute_one_owner(
        task=task,
        factories=factories.as_mapping(),
    )


def _resolve_worker_target(target: str) -> Callable[..., Any]:
    if _TARGET.fullmatch(str(target)) is None:
        raise ValueError("process owner worker target is malformed")
    module_name, attribute = str(target).split(":", 1)
    resolved = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(resolved):
        raise TypeError("process owner worker target is not callable")
    return resolved


def _gpu_id(resource_name: str) -> int | None:
    value = str(resource_name)
    if value == "cpu":
        return None
    if not value.startswith("cuda:"):
        raise ValueError("owner resource must be cpu or one explicit cuda:N")
    suffix = value.split(":", 1)[1]
    if not suffix.isdigit():
        raise ValueError("owner CUDA resource has a nonnumeric index")
    return int(suffix)


def _task_execution_resources(
    task: RoleNeutralPhysicalOwnerTask,
) -> tuple[str, ...]:
    """All resources that must be reserved while one owner is executing."""

    topology = task.neural_query_execution_topology
    if not isinstance(topology, NeuralQueryExecutionTopology):
        raise TypeError(
            "owner task lacks its typed neural-query execution topology"
        )
    if topology.primary_device != task.resource:
        raise ValueError(
            "owner task neural-query topology changed its primary resource"
        )
    htr_devices = tuple(str(value) for value in task.htr_fold_devices)
    if not htr_devices or task.resource not in htr_devices:
        raise ValueError(
            "owner task HTR fold resources omit its primary resource"
        )
    resources = tuple(
        dict.fromkeys((*topology.devices, *htr_devices))
    )
    for device in resources:
        _gpu_id(device)
    return resources


def _runtime_neural_query_topology_attestation(
    topology: NeuralQueryExecutionTopology,
    *,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Prove availability and exact accelerator homogeneity in the worker."""

    if not isinstance(topology, NeuralQueryExecutionTopology):
        raise TypeError(
            "runtime topology validation requires a typed neural-query topology"
        )
    if topology.devices == ("cpu",):
        return {
            "schema_version": (
                "neural_query_runtime_device_topology_attestation_v1"
            ),
            "devices": ["cpu"],
            "backend": "cpu",
            "homogeneous": True,
            "scientific_identity_includes_topology": False,
        }
    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    cuda = getattr(torch, "cuda", None)
    if (
        cuda is None
        or not callable(getattr(cuda, "is_available", None))
        or not bool(cuda.is_available())
        or not callable(getattr(cuda, "device_count", None))
        or not callable(getattr(cuda, "get_device_properties", None))
    ):
        raise RuntimeError(
            "requested neural-query accelerator topology is unavailable"
        )
    count = int(cuda.device_count())
    signatures: list[dict[str, Any]] = []
    for device in topology.devices:
        gpu_id = _gpu_id(device)
        if gpu_id is None or gpu_id >= count:
            raise RuntimeError(
                "requested neural-query accelerator topology is unavailable; "
                f"requested={device}, visible_device_count={count}"
            )
        properties = cuda.get_device_properties(gpu_id)
        try:
            signature = {
                "name": str(properties.name),
                "compute_capability_major": int(properties.major),
                "compute_capability_minor": int(properties.minor),
                "total_memory_bytes": int(properties.total_memory),
                "multiprocessor_count": int(properties.multi_processor_count),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "accelerator runtime omitted compatibility properties"
            ) from exc
        if (
            not signature["name"]
            or signature["compute_capability_major"] < 0
            or signature["compute_capability_minor"] < 0
            or signature["total_memory_bytes"] < 1
            or signature["multiprocessor_count"] < 1
        ):
            raise RuntimeError(
                "accelerator runtime reported invalid compatibility properties"
            )
        signatures.append(signature)
    first = signatures[0]
    if any(signature != first for signature in signatures[1:]):
        raise RuntimeError(
            "heterogeneous accelerator resources cannot span one "
            "neural-query context"
        )
    return {
        "schema_version": (
            "neural_query_runtime_device_topology_attestation_v1"
        ),
        "devices": list(topology.devices),
        "backend": "cuda",
        "homogeneous": True,
        "compatibility_signature": first,
        "scientific_identity_includes_topology": False,
    }


def _resource_tuple_has_capacity(
    resources: Sequence[str],
    *,
    active_by_resource: Mapping[str, int],
    maximum_per_resource: int,
) -> bool:
    """Return whether every member of one atomic reservation is available."""

    if isinstance(resources, (str, bytes)):
        raise TypeError("resource reservation requires one device sequence")
    requested = tuple(str(value) for value in resources)
    if isinstance(maximum_per_resource, bool):
        raise TypeError("maximum resource concurrency must be an integer")
    maximum = int(maximum_per_resource)
    if (
        not requested
        or len(requested) != len(set(requested))
        or maximum < 1
        or any(int(value) < 0 for value in active_by_resource.values())
    ):
        raise ValueError("resource reservation ledger is invalid")
    return all(
        int(active_by_resource.get(resource_name, 0)) < maximum
        for resource_name in requested
    )


def _change_resource_tuple_reservation(
    resources: Sequence[str],
    *,
    active_by_resource: dict[str, int],
    delta: int,
) -> None:
    """Atomically add or remove one complete resource-tuple reservation."""

    if isinstance(resources, (str, bytes)):
        raise TypeError("resource reservation requires one device sequence")
    requested = tuple(str(value) for value in resources)
    if isinstance(delta, bool):
        raise TypeError("resource reservation delta must be an integer")
    change = int(delta)
    if (
        not requested
        or len(requested) != len(set(requested))
        or change not in {-1, 1}
    ):
        raise ValueError("resource reservation change is invalid")
    updated = {
        resource_name: int(
            active_by_resource.get(resource_name, 0)
        )
        + change
        for resource_name in requested
    }
    if any(value < 0 for value in updated.values()):
        raise RuntimeError("resource reservation ledger underflowed")
    for resource_name, value in updated.items():
        active_by_resource[resource_name] = value


def _native_thread_environment(thread_count: int) -> dict[str, str]:
    count = int(thread_count)
    if count < 1:
        raise ValueError("native thread count must be positive")
    return {
        **{key: str(count) for key in _NATIVE_THREAD_ENVIRONMENT},
        "TOKENIZERS_PARALLELISM": "false",
    }


def _process_io_counters() -> dict[str, int] | None:
    """Read Linux per-process I/O counters without treating them as artifacts."""

    path = Path("/proc/self/io")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    parsed: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            return None
        value = raw.strip()
        if not value.isdigit():
            return None
        parsed[key.strip()] = int(value)
    required = {"rchar", "wchar", "read_bytes", "write_bytes"}
    if not required.issubset(parsed):
        return None
    return {key: parsed[key] for key in sorted(required)}


def _process_io_delta(
    before: Mapping[str, int] | None,
    after: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None or set(before) != set(after):
        return None
    return {
        key: max(0, int(after[key]) - int(before[key]))
        for key in sorted(before)
    }


def _start_process(
    process: mp.Process,
    *,
    scope_seed: int,
    native_threads: int,
) -> None:
    """Start one child under a globally serialized temporary environment."""

    seed = int(scope_seed)
    if not 0 <= seed < 2**31:
        raise ValueError("scope hash seed must be a nonnegative 31-bit integer")
    replacements = {
        "PYTHONHASHSEED": str(seed),
        **_native_thread_environment(native_threads),
    }
    with _PROCESS_START_ENVIRONMENT_LOCK:
        previous = {key: os.environ.get(key) for key in replacements}
        try:
            os.environ.update(replacements)
            process.start()
        finally:
            for key, prior in previous.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior


def _spawned_owner_entry(
    task: RoleNeutralPhysicalOwnerTask,
    worker_target: str,
    worker_parameters: Mapping[str, Any],
    native_threads: int,
    marker_path: str,
    connection: Any,
) -> None:
    """Child entry: isolate RNG/native pools, execute, then emit one result."""

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    io_before = _process_io_counters()
    try:
        os.environ.update(_native_thread_environment(native_threads))
        _establish_worker_process_group(marker_path)
        determinism_before = _enforce_stage1_torch_determinism()
        topology = task.neural_query_execution_topology
        if not isinstance(topology, NeuralQueryExecutionTopology):
            raise TypeError(
                "spawned owner lacks its typed neural-query topology"
            )
        gpu_id = _gpu_id(task.resource)
        import torch
        from threadpoolctl import threadpool_limits

        topology_attestation = _runtime_neural_query_topology_attestation(
            topology,
            torch_module=torch,
        )
        seed_stage1_scope_rngs(
            task.physical_owner.scope_seed,
            gpu_id=gpu_id,
        )
        execution_gpu_ids = tuple(
            value
            for value in (
                _gpu_id(device)
                for device in _task_execution_resources(task)
            )
            if value is not None
        )
        for execution_gpu_id in execution_gpu_ids:
            torch.cuda.reset_peak_memory_stats(execution_gpu_id)

        torch.set_num_threads(int(native_threads))
        try:
            torch.set_num_interop_threads(int(native_threads))
        except RuntimeError:
            # The child remains process-isolated and all native libraries are
            # still bounded by threadpoolctl and the inherited environment.
            pass
        target = _resolve_worker_target(worker_target)
        with threadpool_limits(limits=int(native_threads)):
            result = target(
                task=task,
                worker_parameters=copy.deepcopy(dict(worker_parameters)),
            )
        if not isinstance(result, RoleNeutralPhysicalOwnerResult):
            raise TypeError(
                "process owner worker returned an untyped physical-owner result"
            )
        determinism_after = _observe_stage1_torch_determinism()
        stable_before = {
            key: value
            for key, value in determinism_before.items()
            if key not in {"torch_version", "cuda_runtime_version"}
        }
        stable_after = {
            key: value
            for key, value in determinism_after.items()
            if key not in {"torch_version", "cuda_runtime_version"}
        }
        if (
            determinism_after.get("policy_active") is not True
            or stable_before != stable_after
        ):
            raise RuntimeError(
                "process owner worker weakened the Torch determinism policy"
            )
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
        io_after = _process_io_counters()
        peak_allocated_by_device = {
            f"cuda:{topology_gpu_id}": int(
                torch.cuda.max_memory_allocated(topology_gpu_id)
            )
            for topology_gpu_id in execution_gpu_ids
        }
        peak_reserved_by_device = {
            f"cuda:{topology_gpu_id}": int(
                torch.cuda.max_memory_reserved(topology_gpu_id)
            )
            for topology_gpu_id in execution_gpu_ids
        }
        peak_allocated = (
            None
            if gpu_id is None
            else peak_allocated_by_device[task.resource]
        )
        peak_reserved = (
            None
            if gpu_id is None
            else peak_reserved_by_device[task.resource]
        )
        telemetry = {
            "schema_version": (
                "production_role_neutral_process_owner_telemetry_v1"
            ),
            "pid": int(os.getpid()),
            "scope_seed": int(task.physical_owner.scope_seed),
            "resource": task.resource,
            "reserved_resources": list(
                _task_execution_resources(task)
            ),
            "neural_query_device_topology": topology_attestation,
            "native_threads": int(native_threads),
            "wall_seconds": max(0.0, time.monotonic() - started_wall),
            "cpu_seconds": max(0.0, time.process_time() - started_cpu),
            "filesystem_input_blocks": max(
                0,
                int(usage_after.ru_inblock - usage_before.ru_inblock),
            ),
            "filesystem_output_blocks": max(
                0,
                int(usage_after.ru_oublock - usage_before.ru_oublock),
            ),
            "process_io_deltas": _process_io_delta(
                io_before,
                io_after,
            ),
            "peak_resident_kib": max(0, int(usage_after.ru_maxrss)),
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "peak_gpu_allocated_bytes_by_device": (
                peak_allocated_by_device
            ),
            "peak_gpu_reserved_bytes_by_device": (
                peak_reserved_by_device
            ),
            "torch_determinism_observed": determinism_after,
            "worker_report": (
                None
                if result.execution_telemetry is None
                else _json_copy(
                    result.execution_telemetry,
                    label="process owner worker report",
                )
            ),
        }
        connection.send(
            {
                "status": "completed",
                "result": result,
                "telemetry": telemetry,
            }
        )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            raise
    finally:
        connection.close()


@dataclass
class _ActiveOwner:
    task: RoleNeutralPhysicalOwnerTask
    process: mp.Process
    connection: Any
    marker_path: Path
    message: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProcessIsolatedRoleNeutralPhysicalOwnerExecutor:
    """Deterministic spawn-only executor with one process per physical owner."""

    max_workers_per_resource: int
    worker_target: str = PRODUCTION_PROCESS_WORKER_TARGET
    worker_parameters: Mapping[str, Any] | None = None
    production_worker_required: bool = True
    poll_interval_seconds: float = 0.05

    # Capability marker consumed by the parent coordinator so the productive
    # canary cannot accidentally execute in its RNG-sharing process.
    process_isolated_physical_owners: bool = field(
        default=True,
        init=False,
    )

    def __post_init__(self) -> None:
        workers = int(self.max_workers_per_resource)
        interval = float(self.poll_interval_seconds)
        if workers < 1:
            raise ValueError("max_workers_per_resource must be positive")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("process executor poll interval must be positive")
        target = str(self.worker_target)
        if _TARGET.fullmatch(target) is None:
            raise ValueError("process executor worker target is malformed")
        if self.production_worker_required and target != PRODUCTION_PROCESS_WORKER_TARGET:
            raise ValueError(
                "production process executor cannot substitute its worker target"
            )
        parameters = (
            None
            if self.worker_parameters is None
            else _json_copy(
                self.worker_parameters,
                label="process executor worker parameters",
            )
        )
        if self.production_worker_required and parameters is not None:
            if frozenset(parameters) != frozenset(
                {"prepared_context_manifest_path"}
            ):
                raise ValueError(
                    "production process executor parameters are not one sealed "
                    "prepared context"
                )
            from .prepared_stage1_context import (
                load_prepared_stage1_context,
            )

            load_prepared_stage1_context(
                Path(
                    str(
                        parameters[
                            "prepared_context_manifest_path"
                        ]
                    )
                )
            )
        object.__setattr__(self, "max_workers_per_resource", workers)
        object.__setattr__(self, "poll_interval_seconds", interval)
        object.__setattr__(self, "worker_target", target)
        object.__setattr__(self, "worker_parameters", parameters)

    def bind_prepared(
        self,
        *,
        prepared: Any,
        producer_factories_builder: Any,
    ) -> "ProcessIsolatedRoleNeutralPhysicalOwnerExecutor":
        if not self.production_worker_required:
            raise ValueError("test process executors cannot bind production inputs")
        if self.worker_parameters is not None:
            raise RuntimeError("process executor is already bound")
        from .prepared_stage1_context import seal_prepared_stage1_context

        artifact = seal_prepared_stage1_context(
            root=(
                Path(prepared.output_path)
                / "sealed_prepared_stage1_context"
            ).resolve(),
            prepared=prepared,
            producer_factories_builder=producer_factories_builder,
        )
        return dataclasses.replace(
            self,
            worker_parameters={
                "prepared_context_manifest_path": str(
                    artifact.manifest_path
                )
            },
        )

    def bind_context(
        self,
        manifest_path: Path | str,
    ) -> "ProcessIsolatedRoleNeutralPhysicalOwnerExecutor":
        """Bind one sealed context for explicit fresh-per-fit execution."""

        if not self.production_worker_required:
            raise ValueError("test process executors cannot bind production inputs")
        if self.worker_parameters is not None:
            raise RuntimeError("process executor is already bound")
        from .prepared_stage1_context import load_prepared_stage1_context

        artifact = load_prepared_stage1_context(Path(manifest_path))
        return dataclasses.replace(
            self,
            worker_parameters={
                "prepared_context_manifest_path": str(
                    artifact.manifest_path
                )
            },
        )

    def _parameters(self) -> Mapping[str, Any]:
        if self.worker_parameters is None:
            raise RuntimeError(
                "process executor must be bound to an authenticated prepared "
                "context before execution"
            )
        return copy.deepcopy(dict(self.worker_parameters))

    def execute(
        self,
        *,
        tasks: Sequence[RoleNeutralPhysicalOwnerTask],
        worker: Callable[
            [RoleNeutralPhysicalOwnerTask],
            RoleNeutralPhysicalOwnerResult,
        ],
        max_workers: int,
        cpu_budget: int,
    ) -> Sequence[RoleNeutralPhysicalOwnerResult]:
        """Execute tasks without invoking the unsafe in-parent worker closure."""

        if not callable(worker):
            raise TypeError("process executor requires the coordinator worker guard")
        rows = tuple(tasks)
        if not rows:
            return ()
        workers = int(max_workers)
        budget = int(cpu_budget)
        if workers < 1 or budget < 1 or workers > budget:
            raise ValueError(
                "process executor requires 1 <= max_workers <= cpu_budget"
            )
        if len({task.component_parent for task in rows}) != len(rows):
            raise ValueError("process executor task roots are duplicated")
        if len({task.physical_owner.scope_id for task in rows}) != len(rows):
            raise ValueError(
                "one process-executor call cannot duplicate physical owners"
            )
        for task in rows:
            if not isinstance(task, RoleNeutralPhysicalOwnerTask):
                raise TypeError("process executor received an untyped owner task")
            _task_execution_resources(task)

        parameters = self._parameters()
        context = mp.get_context("spawn")
        pending = list(rows)
        active: list[_ActiveOwner] = []
        completed: list[RoleNeutralPhysicalOwnerResult] = []
        active_by_resource: dict[str, int] = {}
        maximum_active = min(
            workers,
            len(rows),
        )
        native_threads = max(1, budget // maximum_active)
        failure: BaseException | None = None

        def start_eligible() -> None:
            nonlocal pending
            index = 0
            while len(active) < maximum_active and index < len(pending):
                task = pending[index]
                reserved = _task_execution_resources(task)
                if not _resource_tuple_has_capacity(
                    reserved,
                    active_by_resource=active_by_resource,
                    maximum_per_resource=(
                        self.max_workers_per_resource
                    ),
                ):
                    index += 1
                    continue
                pending.pop(index)
                receive, send = context.Pipe(duplex=False)
                marker = (
                    task.component_parent.parent
                    / (
                        ".process-group-"
                        f"{task.physical_owner.scope_id}-"
                        f"{hashlib.sha256(str(task.component_parent).encode()).hexdigest()[:16]}"
                        ".json"
                    )
                )
                if marker.exists() or marker.is_symlink():
                    receive.close()
                    send.close()
                    raise FileExistsError(
                        "process owner group marker must be fresh"
                    )
                process = context.Process(
                    target=_spawned_owner_entry,
                    args=(
                        task,
                        self.worker_target,
                        parameters,
                        native_threads,
                        str(marker),
                        send,
                    ),
                    name=f"role-neutral-{task.physical_owner.scope_id}",
                )
                _start_process(
                    process,
                    scope_seed=task.physical_owner.scope_seed,
                    native_threads=native_threads,
                )
                send.close()
                active.append(
                    _ActiveOwner(
                        task=task,
                        process=process,
                        connection=receive,
                        marker_path=marker,
                    )
                )
                _change_resource_tuple_reservation(
                    reserved,
                    active_by_resource=active_by_resource,
                    delta=1,
                )

        try:
            while pending or active:
                start_eligible()
                made_progress = False
                for state in tuple(active):
                    if state.message is None and state.connection.poll():
                        message = state.connection.recv()
                        if not isinstance(message, Mapping):
                            raise RuntimeError(
                                "spawned owner sent a malformed IPC result"
                            )
                        state.message = copy.deepcopy(dict(message))
                        made_progress = True
                    if state.process.is_alive():
                        continue
                    state.process.join()
                    if state.message is None and state.connection.poll():
                        message = state.connection.recv()
                        if isinstance(message, Mapping):
                            state.message = copy.deepcopy(dict(message))
                    state.connection.close()
                    active.remove(state)
                    _change_resource_tuple_reservation(
                        _task_execution_resources(state.task),
                        active_by_resource=active_by_resource,
                        delta=-1,
                    )
                    try:
                        if state.marker_path.exists():
                            state.marker_path.unlink()
                    except OSError as exc:
                        raise RuntimeError(
                            "could not remove owned process-group marker"
                        ) from exc
                    message = state.message
                    if (
                        state.process.exitcode != 0
                        or not isinstance(message, Mapping)
                        or message.get("status") != "completed"
                        or not isinstance(
                            message.get("result"),
                            RoleNeutralPhysicalOwnerResult,
                        )
                    ):
                        detail = (
                            "spawned owner emitted no authenticated failure"
                            if not isinstance(message, Mapping)
                            else (
                                f"{message.get('exception_type', 'WorkerError')}: "
                                f"{message.get('message', 'unknown failure')}\n"
                                f"{message.get('traceback', '')}"
                            )
                        )
                        raise RuntimeError(
                            f"spawned role-neutral owner "
                            f"{state.task.physical_owner.scope_id} failed: {detail}"
                        )
                    result = message["result"]
                    telemetry = message.get("telemetry")
                    if not isinstance(telemetry, Mapping):
                        raise RuntimeError(
                            "spawned owner omitted operational telemetry"
                        )
                    completed.append(
                        dataclasses.replace(
                            result,
                            execution_telemetry=copy.deepcopy(dict(telemetry)),
                        )
                    )
                    made_progress = True
                if not made_progress and active:
                    time.sleep(self.poll_interval_seconds)
        except BaseException as exc:
            failure = exc
        finally:
            cleanup_errors: list[BaseException] = []
            for state in tuple(active):
                try:
                    _terminate_process_and_descendants(
                        state.process,
                        process_group_marker_path=state.marker_path,
                    )
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                finally:
                    state.connection.close()
                    try:
                        if state.marker_path.exists():
                            state.marker_path.unlink()
                    except OSError as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
            if failure is None and cleanup_errors:
                failure = RuntimeError(
                    "process executor could not clean an owned worker group"
                )
        if failure is not None:
            raise failure
        return tuple(completed)


__all__ = [
    "PROCESS_ISOLATED_ROLE_NEUTRAL_AUTHORITY_SCHEMA",
    "PROCESS_ISOLATED_ROLE_NEUTRAL_EXECUTOR_SCHEMA",
    "PRODUCTION_PROCESS_WORKER_TARGET",
    "PreparedRoleNeutralProcessAuthority",
    "ProcessIsolatedRoleNeutralPhysicalOwnerExecutor",
    "_change_resource_tuple_reservation",
    "_resource_tuple_has_capacity",
    "_runtime_neural_query_topology_attestation",
    "_task_execution_resources",
]
