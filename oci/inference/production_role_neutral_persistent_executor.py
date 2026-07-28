"""Persistent spawn-slot executor for role-neutral Stage 1 physical owners.

One resource-bound spawned process authenticates one
``PreparedStage1ContextArtifact`` and then executes owners sequentially.  The
slot is short-lived at the workflow-executor boundary, but preparation cost is
constant in the number of owners assigned to that slot.  Every owner receives
an explicit RNG reset and fair share of the host CPU/native-thread budget.

The existing fresh-process-per-owner executor remains available only as an
explicit operational benchmark mode.  Production, the productive canary, and
the real benchmark use this persistent implementation by default.
"""

from __future__ import annotations

import copy
import concurrent.futures
import dataclasses
import gc
import hashlib
import math
import multiprocessing as mp
import os
import resource
import shutil
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .prepared_stage1_context import (
    PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
    load_prepared_stage1_context,
    seal_prepared_stage1_context,
)
from .production_role_neutral_process_executor import (
    PRODUCTION_PROCESS_WORKER_TARGET,
    _gpu_id,
    _json_copy,
    _native_thread_environment,
    _plan_is_exact_projection,
    _process_io_counters,
    _process_io_delta,
    _resolve_worker_target,
    _runtime_neural_query_topology_attestation,
    _start_process,
    _task_execution_resources,
)
from .production_stage1_role_neutral_execution import (
    RoleNeutralPhysicalOwnerResult,
    RoleNeutralPhysicalOwnerTask,
    _execute_one_owner,
)
from .production_stage1_scope_scheduler import (
    _enforce_stage1_torch_determinism,
    _establish_worker_process_group,
    _observe_stage1_torch_determinism,
    _terminate_process_and_descendants,
    seed_stage1_scope_rngs,
)


PERSISTENT_ROLE_NEUTRAL_EXECUTOR_SCHEMA = (
    "production_role_neutral_persistent_spawn_slot_executor_v1"
)
PERSISTENT_ROLE_NEUTRAL_WORKER_MODE = "persistent_spawn_slots_v1"
FRESH_ROLE_NEUTRAL_WORKER_MODE = "fresh_spawn_per_owner_v1"
PERSISTENT_CONTEXT_PARAMETERS = frozenset(
    {"prepared_context_manifest_path"}
)


def _stable_determinism_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key not in {"torch_version", "cuda_runtime_version"}
    }


def _slot_marker(
    *,
    task: RoleNeutralPhysicalOwnerTask,
    resource_name: str,
    slot_index: int,
) -> Path:
    binding = (
        f"{task.component_parent.parent}|{resource_name}|{int(slot_index)}"
    )
    suffix = hashlib.sha256(binding.encode("utf-8")).hexdigest()[:16]
    return (
        task.component_parent.parent
        / f".persistent-process-group-slot-{int(slot_index)}-{suffix}.json"
    )


def _persistent_slot_entry(
    *,
    resource_name: str,
    slot_index: int,
    worker_target: str,
    worker_parameters: Mapping[str, Any],
    production_worker_required: bool,
    slot_cpu_budget: int,
    host_cpu_budget: int,
    active_slot_count: int,
    initial_scope_seed: int,
    marker_path: str,
    connection: Any,
) -> None:
    """Authenticate once, then consume sequential owner commands."""

    slot_started_wall = time.monotonic()
    slot_started_cpu = time.process_time()
    slot_usage_before = resource.getrusage(resource.RUSAGE_SELF)
    slot_io_before = _process_io_counters()
    completed_count = 0
    try:
        os.environ.update(_native_thread_environment(slot_cpu_budget))
        _establish_worker_process_group(marker_path)
        determinism_at_start = _enforce_stage1_torch_determinism()
        gpu_id = _gpu_id(resource_name)
        seed_stage1_scope_rngs(initial_scope_seed, gpu_id=gpu_id)
        import torch
        from threadpoolctl import threadpool_limits

        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        torch.set_num_threads(int(slot_cpu_budget))
        try:
            torch.set_num_interop_threads(int(slot_cpu_budget))
        except RuntimeError:
            pass

        prepared = None
        factories = None
        context = None
        target = None
        if production_worker_required:
            if set(worker_parameters) != PERSISTENT_CONTEXT_PARAMETERS:
                raise ValueError(
                    "persistent production worker parameters are not closed"
                )
            context = load_prepared_stage1_context(
                Path(
                    str(
                        worker_parameters[
                            "prepared_context_manifest_path"
                        ]
                    )
                )
            )
            prepared, factories = context.reconstruct(
                slot_cpu_budget=slot_cpu_budget,
            )
        else:
            target = _resolve_worker_target(worker_target)
        context_loaded_wall = time.monotonic()
        context_loaded_cpu = time.process_time()
        context_usage_after = resource.getrusage(resource.RUSAGE_SELF)
        context_io_after = _process_io_counters()
        connection.send(
            {
                "status": "ready",
                "slot_index": int(slot_index),
                "pid": int(os.getpid()),
                "resource": resource_name,
                "slot_cpu_budget": int(slot_cpu_budget),
                "host_cpu_budget": int(host_cpu_budget),
                "active_slot_count": int(active_slot_count),
                "prepared_context_load_count": (
                    1 if production_worker_required else 0
                ),
                "prepared_context_content_root_sha256": (
                    None if context is None else context.content_root_sha256
                ),
                "prepared_context_load_wall_seconds": max(
                    0.0,
                    context_loaded_wall - slot_started_wall,
                ),
                "prepared_context_load_cpu_seconds": max(
                    0.0,
                    context_loaded_cpu - slot_started_cpu,
                ),
                "prepared_context_process_io_deltas": _process_io_delta(
                    slot_io_before,
                    context_io_after,
                ),
                "prepared_context_filesystem_input_blocks": max(
                    0,
                    int(
                        context_usage_after.ru_inblock
                        - slot_usage_before.ru_inblock
                    ),
                ),
            }
        )

        while True:
            command = connection.recv()
            if not isinstance(command, Mapping):
                raise ValueError("persistent slot received a malformed command")
            if command.get("command") == "shutdown":
                if set(command) != {"command"}:
                    raise ValueError(
                        "persistent slot shutdown command is not closed"
                    )
                break
            if set(command) != {"command", "task"} or command.get(
                "command"
            ) != "execute":
                raise ValueError("persistent slot received an unknown command")
            task = command["task"]
            if not isinstance(task, RoleNeutralPhysicalOwnerTask):
                raise TypeError("persistent slot received an untyped owner task")
            if task.resource != resource_name:
                raise ValueError(
                    "persistent slot received an owner for another resource"
                )
            if prepared is not None and not _plan_is_exact_projection(
                prepared.stage1_scope_plan,
                task.plan,
            ):
                raise ValueError(
                    "persistent owner task is not an exact prepared-plan "
                    "projection"
                )

            owner_started_wall = time.monotonic()
            owner_started_cpu = time.process_time()
            owner_usage_before = resource.getrusage(resource.RUSAGE_SELF)
            # Charge the one-time preparation read to the first owner in this
            # slot. Summing owner telemetry therefore counts it exactly once.
            owner_io_before = (
                slot_io_before
                if completed_count == 0
                else _process_io_counters()
            )
            topology_attestation = (
                _runtime_neural_query_topology_attestation(
                    task.neural_query_execution_topology,
                    torch_module=torch,
                )
            )
            seed_stage1_scope_rngs(
                task.physical_owner.scope_seed,
                gpu_id=gpu_id,
            )
            topology_gpu_ids = tuple(
                value
                for value in (
                    _gpu_id(device)
                    for device in _task_execution_resources(task)
                )
                if value is not None
            )
            for topology_gpu_id in topology_gpu_ids:
                torch.cuda.reset_peak_memory_stats(topology_gpu_id)
            with threadpool_limits(limits=int(slot_cpu_budget)):
                if factories is not None:
                    result = _execute_one_owner(
                        task=task,
                        factories=factories.as_mapping(),
                    )
                else:
                    assert target is not None
                    result = target(
                        task=task,
                        worker_parameters=copy.deepcopy(
                            dict(worker_parameters)
                        ),
                    )
            if not isinstance(result, RoleNeutralPhysicalOwnerResult):
                raise TypeError(
                    "persistent slot returned an untyped owner result"
                )
            determinism_after = _observe_stage1_torch_determinism()
            if (
                determinism_after.get("policy_active") is not True
                or _stable_determinism_projection(determinism_after)
                != _stable_determinism_projection(determinism_at_start)
            ):
                raise RuntimeError(
                    "persistent slot weakened the Torch determinism policy"
                )
            owner_usage_after = resource.getrusage(resource.RUSAGE_SELF)
            owner_io_after = _process_io_counters()
            peak_allocated_by_device = {
                f"cuda:{topology_gpu_id}": int(
                    torch.cuda.max_memory_allocated(topology_gpu_id)
                )
                for topology_gpu_id in topology_gpu_ids
            }
            peak_reserved_by_device = {
                f"cuda:{topology_gpu_id}": int(
                    torch.cuda.max_memory_reserved(topology_gpu_id)
                )
                for topology_gpu_id in topology_gpu_ids
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
            completed_count += 1
            telemetry = {
                "schema_version": (
                    "production_role_neutral_process_owner_telemetry_v1"
                ),
                "executor_schema_version": (
                    PERSISTENT_ROLE_NEUTRAL_EXECUTOR_SCHEMA
                ),
                "worker_lifecycle_mode": (
                    PERSISTENT_ROLE_NEUTRAL_WORKER_MODE
                ),
                "pid": int(os.getpid()),
                "slot_index": int(slot_index),
                "slot_owner_ordinal": int(completed_count),
                "slot_startup_python_hash_seed": int(initial_scope_seed),
                "per_owner_python_hash_secret_reset_claimed": False,
                "scope_seed": int(task.physical_owner.scope_seed),
                "resource": task.resource,
                "reserved_resources": list(
                    _task_execution_resources(task)
                ),
                "neural_query_device_topology": (
                    topology_attestation
                ),
                "native_threads": int(slot_cpu_budget),
                "slot_cpu_budget": int(slot_cpu_budget),
                "host_cpu_budget": int(host_cpu_budget),
                "active_slot_count": int(active_slot_count),
                "prepared_context_load_count": (
                    1 if production_worker_required else 0
                ),
                "prepared_context_load_charged_to_this_owner": (
                    completed_count == 1
                ),
                "prepared_context_content_root_sha256": (
                    None if context is None else context.content_root_sha256
                ),
                "wall_seconds": max(
                    0.0,
                    time.monotonic() - owner_started_wall,
                ),
                "cpu_seconds": max(
                    0.0,
                    time.process_time() - owner_started_cpu,
                ),
                "filesystem_input_blocks": max(
                    0,
                    int(
                        owner_usage_after.ru_inblock
                        - (
                            slot_usage_before.ru_inblock
                            if completed_count == 1
                            else owner_usage_before.ru_inblock
                        )
                    ),
                ),
                "filesystem_output_blocks": max(
                    0,
                    int(
                        owner_usage_after.ru_oublock
                        - (
                            slot_usage_before.ru_oublock
                            if completed_count == 1
                            else owner_usage_before.ru_oublock
                        )
                    ),
                ),
                "process_io_deltas": _process_io_delta(
                    owner_io_before,
                    owner_io_after,
                ),
                "peak_resident_kib": max(
                    0,
                    int(owner_usage_after.ru_maxrss),
                ),
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
                        label="persistent owner worker report",
                    )
                ),
            }
            connection.send(
                {
                    "status": "completed",
                    "slot_index": int(slot_index),
                    "result": result,
                    "telemetry": telemetry,
                }
            )
            # No fitted model, labels, or RNG continuation may leak into the
            # next owner. Producers return only authenticated artifact
            # receipts; collect discarded invocation-local objects now.
            result = None
            gc.collect()
            if gpu_id is not None:
                torch.cuda.empty_cache()
        connection.send(
            {
                "status": "closed",
                "slot_index": int(slot_index),
                "pid": int(os.getpid()),
                "completed_owner_count": int(completed_count),
            }
        )
    except BaseException as exc:
        try:
            connection.send(
                {
                    "status": "failed",
                    "slot_index": int(slot_index),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


@dataclass
class _PersistentSlot:
    index: int
    resource: str
    process: mp.Process
    connection: Any
    marker_path: Path
    current_task: RoleNeutralPhysicalOwnerTask | None = None
    ready: bool = False
    shutting_down: bool = False
    closed_message: bool = False
    busy: bool = False


def _authenticate_persistent_slot_startup(
    *,
    slots: Sequence[_PersistentSlot],
    slot_cpu_budget: int,
    host_cpu_budget: int,
    active_slot_count: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> None:
    """Require every configured slot to attest readiness before dispatch."""

    pending = list(slots)
    if not pending:
        raise ValueError("persistent startup requires at least one slot")
    deadline = time.monotonic() + float(timeout_seconds)
    while pending:
        made_progress = False
        for slot in tuple(pending):
            if slot.connection.poll():
                try:
                    message = slot.connection.recv()
                except EOFError as exc:
                    raise RuntimeError(
                        "persistent slot closed IPC before its startup "
                        "attestation"
                    ) from exc
                if not isinstance(message, Mapping):
                    raise RuntimeError(
                        "persistent slot sent malformed startup IPC"
                    )
                status = message.get("status")
                if status == "failed":
                    raise RuntimeError(
                        "persistent role-neutral slot "
                        f"{slot.index} failed during startup: "
                        f"{message.get('exception_type', 'WorkerError')}: "
                        f"{message.get('message', 'unknown failure')}\n"
                        f"{message.get('traceback', '')}"
                    )
                if status != "ready":
                    raise RuntimeError(
                        "persistent slot sent a non-ready startup "
                        f"message: {status!r}"
                    )
                if slot.ready:
                    raise RuntimeError(
                        "persistent slot sent duplicate ready"
                    )
                if (
                    message.get("slot_index") != slot.index
                    or message.get("resource") != slot.resource
                    or message.get("slot_cpu_budget") != slot_cpu_budget
                    or message.get("host_cpu_budget") != host_cpu_budget
                    or message.get("active_slot_count")
                    != active_slot_count
                ):
                    raise RuntimeError(
                        "persistent slot failed its startup attestation"
                    )
                slot.ready = True
                pending.remove(slot)
                made_progress = True
            elif not slot.process.is_alive():
                slot.process.join()
                raise RuntimeError(
                    "persistent role-neutral slot exited before its startup "
                    "attestation"
                )
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "persistent session slot did not authenticate in time"
            )
        if not made_progress:
            time.sleep(min(float(poll_interval_seconds), remaining))


class PersistentRoleNeutralExecutionSession:
    """Thread-safe persistent slot pool shared across executor calls."""

    def __init__(
        self,
        *,
        executor: "PersistentSpawnRoleNeutralPhysicalOwnerExecutor",
        resources: Sequence[str],
        max_workers: int,
        cpu_budget: int,
        marker_root: Path | str | None,
    ) -> None:
        self._executor = executor
        self._condition = threading.Condition()
        self._closed = False
        self._broken: BaseException | None = None
        self._active_calls = 0
        self._slots: list[_PersistentSlot] = []
        self._parameters = executor._parameters()
        resource_order = tuple(dict.fromkeys(str(value) for value in resources))
        if not resource_order:
            raise ValueError("persistent session requires at least one resource")
        for value in resource_order:
            _gpu_id(value)
        workers = int(max_workers)
        budget = int(cpu_budget)
        if workers < 1 or budget < 1 or workers > budget:
            raise ValueError(
                "persistent session requires 1 <= max_workers <= cpu_budget"
            )
        # ``workers`` is the concurrent-owner cap. One owner may atomically
        # reserve several devices, so the pool still needs at least one
        # process slot per selected resource when that cap is one.
        self.max_parallel_owners = workers
        slots_by_resource = {value: 1 for value in resource_order}
        remaining = max(0, workers - len(resource_order))
        while remaining:
            advanced = False
            for value in resource_order:
                if (
                    slots_by_resource[value]
                    < executor.max_workers_per_resource
                ):
                    slots_by_resource[value] += 1
                    remaining -= 1
                    advanced = True
                    if remaining == 0:
                        break
            if not advanced:
                break
        self.active_slot_count = sum(slots_by_resource.values())
        self.host_cpu_budget = budget
        self.slot_cpu_budget = max(1, budget // self.active_slot_count)
        if marker_root is None:
            if executor.production_worker_required:
                manifest = Path(
                    str(
                        self._parameters[
                            "prepared_context_manifest_path"
                        ]
                    )
                )
                parent = manifest.parent.parent
            else:
                parent = Path(tempfile.gettempdir())
            parent.mkdir(parents=True, exist_ok=True)
            self._marker_root = Path(
                tempfile.mkdtemp(
                    prefix=".role-neutral-persistent-session-",
                    dir=str(parent),
                )
            )
        else:
            target = Path(marker_root)
            if not target.is_absolute():
                raise ValueError(
                    "persistent session marker root must be absolute"
                )
            if target.exists() or target.is_symlink():
                raise FileExistsError(
                    "persistent session marker root must be fresh"
                )
            target.mkdir(parents=True, exist_ok=False)
            self._marker_root = target
        context = mp.get_context("spawn")
        try:
            slot_index = 0
            for resource_name in resource_order:
                for _local_index in range(slots_by_resource[resource_name]):
                    marker = (
                        self._marker_root
                        / f"process-group-slot-{slot_index}.json"
                    )
                    parent_connection, child_connection = context.Pipe(
                        duplex=True
                    )
                    initial_seed = int(
                        hashlib.sha256(
                            (
                                f"{PERSISTENT_ROLE_NEUTRAL_WORKER_MODE}|"
                                f"{resource_name}|{slot_index}"
                            ).encode("utf-8")
                        ).hexdigest()[:7],
                        16,
                    )
                    process = context.Process(
                        target=_persistent_slot_entry,
                        kwargs={
                            "resource_name": resource_name,
                            "slot_index": slot_index,
                            "worker_target": executor.worker_target,
                            "worker_parameters": self._parameters,
                            "production_worker_required": (
                                executor.production_worker_required
                            ),
                            "slot_cpu_budget": self.slot_cpu_budget,
                            "host_cpu_budget": budget,
                            "active_slot_count": self.active_slot_count,
                            "initial_scope_seed": initial_seed,
                            "marker_path": str(marker),
                            "connection": child_connection,
                        },
                        name=(
                            f"role-neutral-session-slot-{slot_index}-"
                            f"{resource_name.replace(':', '-')}"
                        ),
                    )
                    _start_process(
                        process,
                        scope_seed=initial_seed,
                        native_threads=self.slot_cpu_budget,
                    )
                    child_connection.close()
                    slot = _PersistentSlot(
                        index=slot_index,
                        resource=resource_name,
                        process=process,
                        connection=parent_connection,
                        marker_path=marker,
                    )
                    self._slots.append(slot)
                    slot_index += 1
            _authenticate_persistent_slot_startup(
                slots=self._slots,
                slot_cpu_budget=self.slot_cpu_budget,
                host_cpu_budget=budget,
                active_slot_count=self.active_slot_count,
                poll_interval_seconds=executor.poll_interval_seconds,
                timeout_seconds=executor.startup_timeout_seconds,
            )
        except BaseException:
            self._terminate()
            raise

    @property
    def worker_lifecycle_mode(self) -> str:
        return PERSISTENT_ROLE_NEUTRAL_WORKER_MODE

    @property
    def process_isolated_physical_owners(self) -> bool:
        return True

    def _acquire(
        self,
        resource_names: Sequence[str],
    ) -> tuple[_PersistentSlot, ...]:
        requested = tuple(str(value) for value in resource_names)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError(
                "persistent session resource reservation must be nonempty "
                "and duplicate-free"
            )
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("persistent execution session is closed")
                if self._broken is not None:
                    raise RuntimeError(
                        "persistent execution session is broken"
                    ) from self._broken
                missing = [
                    resource_name
                    for resource_name in requested
                    if not any(
                        slot.resource == resource_name
                        for slot in self._slots
                    )
                ]
                if missing:
                    raise ValueError(
                        "persistent session has no slot for one or more "
                        f"reserved task resources: {missing}"
                    )
                selected: list[_PersistentSlot] = []
                for resource_name in requested:
                    available = next(
                        (
                            slot
                            for slot in self._slots
                            if slot.resource == resource_name
                            and not slot.busy
                        ),
                        None,
                    )
                    if available is None:
                        selected = []
                        break
                    selected.append(available)
                if selected:
                    for slot in selected:
                        slot.busy = True
                    self._active_calls += 1
                    return tuple(selected)
                self._condition.wait()

    def _release(
        self,
        slots: Sequence[_PersistentSlot],
        *,
        failure: BaseException | None,
    ) -> None:
        reserved = tuple(slots)
        if not reserved:
            raise ValueError("persistent session cannot release no slots")
        with self._condition:
            for slot in reserved:
                slot.busy = False
            self._active_calls -= 1
            if failure is not None and self._broken is None:
                self._broken = failure
            self._condition.notify_all()

    def _execute_task(
        self,
        task: RoleNeutralPhysicalOwnerTask,
    ) -> RoleNeutralPhysicalOwnerResult:
        reserved = self._acquire(_task_execution_resources(task))
        slot = reserved[0]
        if slot.resource != task.resource:
            self._release(reserved, failure=None)
            raise RuntimeError(
                "persistent session reservation changed the primary resource"
            )
        failure: BaseException | None = None
        try:
            slot.connection.send({"command": "execute", "task": task})
            message = slot.connection.recv()
            if not isinstance(message, Mapping):
                raise RuntimeError(
                    "persistent session slot sent malformed IPC"
                )
            if message.get("status") == "failed":
                raise RuntimeError(
                    f"persistent role-neutral slot {slot.index} failed: "
                    f"{message.get('exception_type', 'WorkerError')}: "
                    f"{message.get('message', 'unknown failure')}\n"
                    f"{message.get('traceback', '')}"
                )
            result = message.get("result")
            telemetry = message.get("telemetry")
            if (
                message.get("status") != "completed"
                or not isinstance(result, RoleNeutralPhysicalOwnerResult)
                or result.physical_owner_scope_id
                != task.physical_owner.scope_id
                or not isinstance(telemetry, Mapping)
            ):
                raise RuntimeError(
                    "persistent session slot substituted its owner result"
                )
            return dataclasses.replace(
                result,
                execution_telemetry=copy.deepcopy(dict(telemetry)),
            )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            self._release(reserved, failure=failure)

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
        if not callable(worker):
            raise TypeError(
                "persistent session requires the coordinator worker guard"
            )
        rows = tuple(tasks)
        if not rows:
            return ()
        if int(cpu_budget) != self.host_cpu_budget:
            raise ValueError(
                "persistent session CPU budget changed after startup"
            )
        requested = int(max_workers)
        if requested < 1 or requested > self.max_parallel_owners:
            raise ValueError(
                "persistent session max_workers exceeds its concurrent-owner "
                "allocation"
            )
        if len({task.component_parent for task in rows}) != len(rows):
            raise ValueError("persistent session task roots are duplicated")
        for task in rows:
            if not isinstance(task, RoleNeutralPhysicalOwnerTask):
                raise TypeError(
                    "persistent session received an untyped owner task"
                )
            _task_execution_resources(task)
        if len(rows) == 1:
            return (self._execute_task(rows[0]),)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(requested, len(rows)),
            thread_name_prefix="role-neutral-persistent-session",
        ) as pool:
            futures = [pool.submit(self._execute_task, task) for task in rows]
            return tuple(future.result() for future in futures)

    def _terminate(self) -> None:
        for slot in tuple(self._slots):
            try:
                _terminate_process_and_descendants(
                    slot.process,
                    process_group_marker_path=slot.marker_path,
                )
            except BaseException:
                pass
            try:
                slot.connection.close()
            except OSError:
                pass
            try:
                if slot.marker_path.exists():
                    slot.marker_path.unlink()
            except OSError:
                pass
        self._slots.clear()
        if getattr(self, "_marker_root", None) is not None:
            shutil.rmtree(self._marker_root, ignore_errors=True)

    def interrupt(self) -> None:
        """Stop every owned worker group without waiting for active calls."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._terminate()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            while self._active_calls:
                self._condition.wait()
            self._closed = True
            self._condition.notify_all()
        failure: BaseException | None = self._broken
        try:
            if failure is None:
                for slot in self._slots:
                    slot.connection.send({"command": "shutdown"})
                    slot.shutting_down = True
                for slot in self._slots:
                    if not slot.connection.poll(60):
                        raise RuntimeError(
                            "persistent session slot did not close in time"
                        )
                    message = slot.connection.recv()
                    if (
                        not isinstance(message, Mapping)
                        or message.get("status") != "closed"
                    ):
                        raise RuntimeError(
                            "persistent session slot omitted terminal closure"
                        )
                    slot.process.join(timeout=60)
                    if slot.process.exitcode != 0:
                        raise RuntimeError(
                            "persistent session slot exited unsuccessfully"
                        )
        except BaseException as exc:
            failure = failure or exc
        finally:
            self._terminate()
        if failure is not None:
            raise failure

    def __enter__(self) -> "PersistentRoleNeutralExecutionSession":
        return self

    def __exit__(self, exc_type, exc, traceback_value) -> bool:
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise
        return False


@dataclass(frozen=True)
class PersistentSpawnRoleNeutralPhysicalOwnerExecutor:
    """Spawn-only executor that prepares once per resource-bound slot."""

    max_workers_per_resource: int
    startup_timeout_seconds: float
    worker_target: str = PRODUCTION_PROCESS_WORKER_TARGET
    worker_parameters: Mapping[str, Any] | None = None
    production_worker_required: bool = True
    poll_interval_seconds: float = 0.05
    worker_lifecycle_mode: str = PERSISTENT_ROLE_NEUTRAL_WORKER_MODE
    process_isolated_physical_owners: bool = field(
        default=True,
        init=False,
    )

    def __post_init__(self) -> None:
        workers = int(self.max_workers_per_resource)
        interval = float(self.poll_interval_seconds)
        startup_timeout = float(self.startup_timeout_seconds)
        if workers < 1:
            raise ValueError("max_workers_per_resource must be positive")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError(
                "persistent executor poll interval must be positive"
            )
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError(
                "persistent executor startup timeout must be positive"
            )
        if self.worker_lifecycle_mode != PERSISTENT_ROLE_NEUTRAL_WORKER_MODE:
            raise ValueError(
                "persistent executor requires its explicit lifecycle mode"
            )
        if (
            self.production_worker_required
            and self.worker_target != PRODUCTION_PROCESS_WORKER_TARGET
        ):
            raise ValueError(
                "persistent production executor cannot substitute its target"
            )
        parameters = (
            None
            if self.worker_parameters is None
            else _json_copy(
                self.worker_parameters,
                label="persistent executor parameters",
            )
        )
        if self.production_worker_required and parameters is not None:
            if set(parameters) != PERSISTENT_CONTEXT_PARAMETERS:
                raise ValueError(
                    "persistent executor context parameters are not closed"
                )
            manifest = Path(
                str(parameters["prepared_context_manifest_path"])
            )
            load_prepared_stage1_context(manifest)
        elif not self.production_worker_required and parameters is None:
            raise ValueError(
                "test persistent executor requires explicit worker parameters"
            )
        object.__setattr__(self, "max_workers_per_resource", workers)
        object.__setattr__(self, "poll_interval_seconds", interval)
        object.__setattr__(
            self,
            "startup_timeout_seconds",
            startup_timeout,
        )
        object.__setattr__(self, "worker_parameters", parameters)

    def bind_prepared(
        self,
        *,
        prepared: Any,
        producer_factories_builder: Any,
    ) -> "PersistentSpawnRoleNeutralPhysicalOwnerExecutor":
        if not self.production_worker_required:
            raise ValueError("test persistent executor cannot bind production")
        if self.worker_parameters is not None:
            raise RuntimeError("persistent executor is already bound")
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
    ) -> "PersistentSpawnRoleNeutralPhysicalOwnerExecutor":
        """Bind an already sealed context without republishing preparation."""

        if not self.production_worker_required:
            raise ValueError("test persistent executor cannot bind production")
        if self.worker_parameters is not None:
            raise RuntimeError("persistent executor is already bound")
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
                "persistent executor must be bound to a prepared context"
            )
        return copy.deepcopy(dict(self.worker_parameters))

    def open_session(
        self,
        *,
        resources: Sequence[str],
        max_workers: int,
        cpu_budget: int,
        marker_root: Path | str | None = None,
    ) -> PersistentRoleNeutralExecutionSession:
        """Start one reusable, explicitly closed slot pool."""

        return PersistentRoleNeutralExecutionSession(
            executor=self,
            resources=resources,
            max_workers=max_workers,
            cpu_budget=cpu_budget,
            marker_root=marker_root,
        )

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
        """Execute all owners, reusing one authenticated context per slot."""

        if not callable(worker):
            raise TypeError(
                "persistent executor requires the coordinator worker guard"
            )
        rows = tuple(tasks)
        if not rows:
            return ()
        workers = int(max_workers)
        budget = int(cpu_budget)
        if workers < 1 or budget < 1 or workers > budget:
            raise ValueError(
                "persistent executor requires 1 <= max_workers <= cpu_budget"
            )
        if len({task.component_parent for task in rows}) != len(rows):
            raise ValueError("persistent executor task roots are duplicated")
        if len({task.physical_owner.scope_id for task in rows}) != len(rows):
            raise ValueError(
                "persistent executor cannot duplicate physical owners"
            )
        for task in rows:
            if not isinstance(task, RoleNeutralPhysicalOwnerTask):
                raise TypeError(
                    "persistent executor received an untyped owner task"
                )
            _task_execution_resources(task)
        if any(
            len(_task_execution_resources(task)) > 1
            for task in rows
        ):
            raise RuntimeError(
                "multi-device neural-query contexts require "
                "open_session() so every participating persistent slot can "
                "be reserved atomically"
            )

        by_resource: dict[str, list[RoleNeutralPhysicalOwnerTask]] = {}
        resource_order: list[str] = []
        for task in rows:
            if task.resource not in by_resource:
                by_resource[task.resource] = []
                resource_order.append(task.resource)
            by_resource[task.resource].append(task)
        desired_by_resource = {
            name: min(
                self.max_workers_per_resource,
                len(by_resource[name]),
            )
            for name in resource_order
        }
        # Allocate at least one slot per used resource, then fill remaining
        # capacity in stable resource order.
        if len(resource_order) > workers:
            raise ValueError(
                "max_workers is smaller than the number of used resources"
            )
        slots_by_resource = {name: 1 for name in resource_order}
        remaining = workers - len(resource_order)
        while remaining:
            advanced = False
            for name in resource_order:
                if slots_by_resource[name] < desired_by_resource[name]:
                    slots_by_resource[name] += 1
                    remaining -= 1
                    advanced = True
                    if not remaining:
                        break
            if not advanced:
                break
        active_slot_count = sum(slots_by_resource.values())
        slot_cpu_budget = max(1, budget // active_slot_count)
        parameters = self._parameters()
        context = mp.get_context("spawn")
        slots: list[_PersistentSlot] = []
        completed: list[RoleNeutralPhysicalOwnerResult] = []
        failure: BaseException | None = None

        try:
            slot_index = 0
            for resource_name in resource_order:
                for local_index in range(slots_by_resource[resource_name]):
                    first_task = by_resource[resource_name][local_index]
                    marker = _slot_marker(
                        task=first_task,
                        resource_name=resource_name,
                        slot_index=slot_index,
                    )
                    if marker.exists() or marker.is_symlink():
                        raise FileExistsError(
                            "persistent process-group marker must be fresh"
                        )
                    parent_connection, child_connection = context.Pipe(
                        duplex=True
                    )
                    process = context.Process(
                        target=_persistent_slot_entry,
                        kwargs={
                            "resource_name": resource_name,
                            "slot_index": slot_index,
                            "worker_target": self.worker_target,
                            "worker_parameters": parameters,
                            "production_worker_required": (
                                self.production_worker_required
                            ),
                            "slot_cpu_budget": slot_cpu_budget,
                            "host_cpu_budget": budget,
                            "active_slot_count": active_slot_count,
                            "initial_scope_seed": (
                                first_task.physical_owner.scope_seed
                            ),
                            "marker_path": str(marker),
                            "connection": child_connection,
                        },
                        name=(
                            f"role-neutral-slot-{slot_index}-"
                            f"{resource_name.replace(':', '-')}"
                        ),
                    )
                    _start_process(
                        process,
                        scope_seed=first_task.physical_owner.scope_seed,
                        native_threads=slot_cpu_budget,
                    )
                    child_connection.close()
                    slots.append(
                        _PersistentSlot(
                            index=slot_index,
                            resource=resource_name,
                            process=process,
                            connection=parent_connection,
                            marker_path=marker,
                        )
                    )
                    slot_index += 1

            # Authenticate every configured slot before dispatching any owner.
            # Without this startup barrier, a fast first slot can complete and
            # consume a second queued owner before a later slot's ``ready``
            # message is observed, silently collapsing requested concurrency.
            _authenticate_persistent_slot_startup(
                slots=slots,
                slot_cpu_budget=slot_cpu_budget,
                host_cpu_budget=budget,
                active_slot_count=active_slot_count,
                poll_interval_seconds=self.poll_interval_seconds,
                timeout_seconds=self.startup_timeout_seconds,
            )

            while slots:
                made_progress = False
                for slot in tuple(slots):
                    while slot.connection.poll():
                        try:
                            message = slot.connection.recv()
                        except EOFError:
                            break
                        if not isinstance(message, Mapping):
                            raise RuntimeError(
                                "persistent slot sent malformed IPC"
                            )
                        status = message.get("status")
                        made_progress = True
                        if status == "ready":
                            if slot.ready:
                                raise RuntimeError(
                                    "persistent slot sent duplicate ready"
                                )
                            slot.ready = True
                        elif status == "completed":
                            result = message.get("result")
                            telemetry = message.get("telemetry")
                            if (
                                slot.current_task is None
                                or not isinstance(
                                    result,
                                    RoleNeutralPhysicalOwnerResult,
                                )
                                or not isinstance(telemetry, Mapping)
                                or result.physical_owner_scope_id
                                != slot.current_task.physical_owner.scope_id
                            ):
                                raise RuntimeError(
                                    "persistent slot substituted its owner result"
                                )
                            completed.append(
                                dataclasses.replace(
                                    result,
                                    execution_telemetry=copy.deepcopy(
                                        dict(telemetry)
                                    ),
                                )
                            )
                            slot.current_task = None
                        elif status == "closed":
                            slot.closed_message = True
                            # The child closes its pipe immediately after this
                            # terminal message. Do not interpret the resulting
                            # EOF readiness as another application message.
                            break
                        elif status == "failed":
                            raise RuntimeError(
                                "persistent role-neutral slot "
                                f"{slot.index} failed: "
                                f"{message.get('exception_type', 'WorkerError')}: "
                                f"{message.get('message', 'unknown failure')}\n"
                                f"{message.get('traceback', '')}"
                            )
                        else:
                            raise RuntimeError(
                                "persistent slot sent an unknown status"
                            )

                    if slot.ready and slot.current_task is None and not slot.shutting_down:
                        queue = by_resource[slot.resource]
                        if queue:
                            task = queue.pop(0)
                            slot.connection.send(
                                {"command": "execute", "task": task}
                            )
                            slot.current_task = task
                        else:
                            slot.connection.send({"command": "shutdown"})
                            slot.shutting_down = True
                        made_progress = True

                    if not slot.process.is_alive():
                        slot.process.join()
                        if (
                            slot.process.exitcode != 0
                            or not slot.closed_message
                            or slot.current_task is not None
                        ):
                            raise RuntimeError(
                                "persistent role-neutral slot exited without "
                                "a complete authenticated shutdown"
                            )
                        slot.connection.close()
                        slots.remove(slot)
                        try:
                            if slot.marker_path.exists():
                                slot.marker_path.unlink()
                        except OSError as exc:
                            raise RuntimeError(
                                "could not remove persistent process-group marker"
                            ) from exc
                        made_progress = True
                if not made_progress and slots:
                    time.sleep(self.poll_interval_seconds)
        except BaseException as exc:
            failure = exc
        finally:
            cleanup_errors: list[BaseException] = []
            for slot in tuple(slots):
                try:
                    _terminate_process_and_descendants(
                        slot.process,
                        process_group_marker_path=slot.marker_path,
                    )
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
                finally:
                    slot.connection.close()
                    try:
                        if slot.marker_path.exists():
                            slot.marker_path.unlink()
                    except OSError as cleanup_exc:
                        cleanup_errors.append(cleanup_exc)
            if failure is None and cleanup_errors:
                failure = RuntimeError(
                    "persistent executor could not clean an owned worker group"
                )
        if failure is not None:
            raise failure
        if len(completed) != len(rows):
            raise RuntimeError(
                "persistent executor omitted or added an owner result"
            )
        return tuple(completed)


__all__ = [
    "FRESH_ROLE_NEUTRAL_WORKER_MODE",
    "PERSISTENT_CONTEXT_PARAMETERS",
    "PERSISTENT_ROLE_NEUTRAL_EXECUTOR_SCHEMA",
    "PERSISTENT_ROLE_NEUTRAL_WORKER_MODE",
    "PersistentRoleNeutralExecutionSession",
    "PersistentSpawnRoleNeutralPhysicalOwnerExecutor",
]
