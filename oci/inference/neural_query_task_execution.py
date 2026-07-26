"""Spawn-safe bounded task execution for learned neural queries.

The executor keeps runtime placement outside scientific results.  Process
tasks receive a small authenticated reference to the one shared embedding
cache and a row authority; they reopen read-only mmap arrays in the child and
cannot request a peer row.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from threadpoolctl import threadpool_limits

from .production_stage1_preflight_scope_inputs import (
    ScopedEmbeddingView,
    _authenticated_cache_identity,
    _build_shared_cache_reference,
    _load_shared_cache,
)
from .production_stage1_scope_scheduler import (
    _enforce_stage1_torch_determinism,
    _observe_stage1_torch_determinism,
    _validate_torch_determinism_observation,
)
from .neural_query_operational_controls import (
    RoleNeutralNeuralQueryTaskResourcePlan,
)


NEURAL_QUERY_AUTHENTICATED_CACHE_REFERENCE_SCHEMA = (
    "production_neural_query_authenticated_cache_task_reference_v1"
)
NEURAL_QUERY_TASK_PHASE_ATTESTATION_SCHEMA = (
    "production_neural_query_task_phase_execution_attestation_v1"
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


def _integer_rows(values: Sequence[Any], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be one row-ID sequence")
    rows: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must contain integers")
        rows.append(int(value))
    if not rows or len(rows) != len(set(rows)) or min(rows) < 0:
        raise ValueError(f"{label} must contain unique nonnegative rows")
    return tuple(rows)


def _physical_cache_and_parent_authority(
    bound_provider: Any,
) -> tuple[Any, tuple[int, ...] | None]:
    cache = getattr(bound_provider, "_cache", None)
    if cache is None:
        raise TypeError(
            "neural-query process execution requires a real authenticated "
            "bound embedding provider"
        )
    parent_authority = getattr(cache, "allowed_row_ids", None)
    physical = getattr(cache, "_cache", cache)
    return (
        physical,
        (
            None
            if parent_authority is None
            else tuple(map(int, parent_authority))
        ),
    )


@dataclass(frozen=True)
class NeuralQueryAuthenticatedCacheReference:
    """Pickle-small cache locator plus the exact owner row authority."""

    shared_cache_reference: Mapping[str, Any]
    allowed_row_ids: tuple[int, ...]
    allowed_row_order_sha256: str
    logical_identity_sha256: str
    schema_version: str = NEURAL_QUERY_AUTHENTICATED_CACHE_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != NEURAL_QUERY_AUTHENTICATED_CACHE_REFERENCE_SCHEMA:
            raise ValueError(
                "unsupported neural-query authenticated cache task reference"
            )
        shared = copy.deepcopy(dict(self.shared_cache_reference))
        body = {
            key: copy.deepcopy(value)
            for key, value in shared.items()
            if key != "content_sha256"
        }
        if (
            shared.get("content_sha256") != _sha256_json(body)
            or not isinstance(shared.get("logical_identity"), Mapping)
        ):
            raise ValueError(
                "neural-query task cache reference is not self-authenticating"
            )
        rows = _integer_rows(
            self.allowed_row_ids,
            label="neural-query task allowed rows",
        )
        if self.allowed_row_order_sha256 != _sha256_json(list(rows)):
            raise ValueError(
                "neural-query task row authority fingerprint changed"
            )
        if self.logical_identity_sha256 != _sha256_json(
            shared["logical_identity"]
        ):
            raise ValueError(
                "neural-query task logical cache identity changed"
            )
        row_count = int(shared["logical_identity"].get("row_count", -1))
        if max(rows) >= row_count:
            raise ValueError(
                "neural-query task row authority exceeds the shared cache"
            )
        object.__setattr__(self, "shared_cache_reference", shared)
        object.__setattr__(self, "allowed_row_ids", rows)

    @classmethod
    def from_bound_provider(
        cls,
        bound_provider: Any,
        *,
        allowed_row_ids: Sequence[int],
    ) -> "NeuralQueryAuthenticatedCacheReference":
        """Capture one owner-scoped capability without copying cache payloads."""

        rows = _integer_rows(
            allowed_row_ids,
            label="neural-query task allowed rows",
        )
        provider_rows = tuple(map(int, getattr(bound_provider, "row_ids", ())))
        if provider_rows != rows:
            raise ValueError(
                "neural-query task cache authority must preserve the bound "
                "owner row order"
            )
        physical, parent_authority = _physical_cache_and_parent_authority(
            bound_provider
        )
        if parent_authority is not None and not set(rows).issubset(
            parent_authority
        ):
            raise ValueError(
                "neural-query task cache authority exceeds its parent view"
            )
        existing = getattr(physical, "_shared_reference", None)
        if isinstance(existing, Mapping):
            shared = copy.deepcopy(dict(existing))
        else:
            identity = _authenticated_cache_identity(physical)
            shared = _build_shared_cache_reference(
                embedding_cache=physical,
                embedding_cache_identity=identity,
                global_embedding_cache_path=physical.cache_dir,
            )
        return cls(
            shared_cache_reference=shared,
            allowed_row_ids=rows,
            allowed_row_order_sha256=_sha256_json(list(rows)),
            logical_identity_sha256=_sha256_json(shared["logical_identity"]),
        )

    def open_bound(
        self,
        *,
        row_ids: Sequence[int],
        texts: Sequence[str],
    ) -> Any:
        """Reopen the mmap cache and bind only an authorized ordered subset."""

        requested = _integer_rows(
            row_ids,
            label="neural-query worker requested rows",
        )
        forbidden = [
            row for row in requested if row not in set(self.allowed_row_ids)
        ]
        if forbidden:
            raise PermissionError(
                "neural-query cache task attempted peer-row access: "
                f"{forbidden[:3]}"
            )
        exact_texts = tuple(texts)
        if (
            len(exact_texts) != len(requested)
            or any(not isinstance(value, str) for value in exact_texts)
        ):
            raise ValueError(
                "neural-query worker requires one exact text per requested row"
            )
        shared = _load_shared_cache(dict(self.shared_cache_reference))
        view = ScopedEmbeddingView(
            shared_cache=shared,
            logical_identity=self.shared_cache_reference[
                "logical_identity"
            ],
            allowed_row_ids=self.allowed_row_ids,
            shared_reference_content_sha256=self.shared_cache_reference[
                "content_sha256"
            ],
        )
        bound = view.bind_spent(requested, exact_texts)
        if tuple(map(int, bound.row_ids)) != requested:
            raise RuntimeError(
                "neural-query worker cache changed requested row order"
            )
        return bound


@dataclass(frozen=True)
class CompletedNeuralQueryTask:
    """Worker result plus deployment-only execution telemetry."""

    value: Any
    device: str
    started_monotonic_ns: int
    finished_monotonic_ns: int
    process_id: int
    thread_id: int
    gpu_peak_allocated_bytes: int | None
    gpu_peak_reserved_bytes: int | None
    torch_determinism_observed: Mapping[str, Any] | None


def _invoke_neural_query_task(
    worker: Callable[[Any, str], Any],
    task: Any,
    device: str,
    *,
    worker_cpu_threads: int,
    process_isolated: bool,
) -> CompletedNeuralQueryTask:
    # The task owns its lease from child invocation onward.  Count
    # determinism, thread-pool, and CUDA-device setup in the interval so
    # short scientific tasks cannot conceal real concurrent occupancy.
    started = time.monotonic_ns()
    determinism_before: Mapping[str, Any] | None = None
    if process_isolated:
        threads = str(int(worker_cpu_threads))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads
        os.environ["OPENBLAS_NUM_THREADS"] = threads
        os.environ["NUMEXPR_NUM_THREADS"] = threads
        determinism_before = _validate_torch_determinism_observation(
            _enforce_stage1_torch_determinism()
        )
        import torch

        torch.set_num_threads(int(worker_cpu_threads))
        if (
            torch.get_num_threads() != int(worker_cpu_threads)
            or torch.get_num_interop_threads() != int(worker_cpu_threads)
        ):
            raise RuntimeError(
                "spawned neural-query worker did not preserve its "
                "one-thread Torch CPU lease"
            )

    peak_allocated: int | None = None
    peak_reserved: int | None = None
    torch_module: Any | None = None
    if str(device).startswith("cuda:"):
        import torch

        torch_module = torch
        if torch.cuda.is_available():
            torch.cuda.set_device(str(device))
            torch.cuda.reset_peak_memory_stats(str(device))
    with threadpool_limits(limits=int(worker_cpu_threads)):
        value = worker(task, str(device))
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(str(device))
        peak_allocated = int(
            torch_module.cuda.max_memory_allocated(str(device))
        )
        peak_reserved = int(
            torch_module.cuda.max_memory_reserved(str(device))
        )
        if (
            peak_allocated <= 0
            or peak_reserved <= 0
            or peak_reserved < peak_allocated
        ):
            raise RuntimeError(
                "CUDA task did not report positive coherent allocated/"
                "reserved Torch peaks"
            )
    finished = time.monotonic_ns()
    if finished <= started:
        raise RuntimeError("neural-query task interval clock did not advance")

    determinism_after: Mapping[str, Any] | None = None
    if process_isolated:
        determinism_after = _validate_torch_determinism_observation(
            _observe_stage1_torch_determinism()
        )
        if dict(determinism_after) != dict(determinism_before or {}):
            raise RuntimeError(
                "neural-query task weakened strict Stage 1 Torch "
                "determinism"
            )
    return CompletedNeuralQueryTask(
        value=value,
        device=str(device),
        started_monotonic_ns=started,
        finished_monotonic_ns=finished,
        process_id=os.getpid(),
        thread_id=threading.get_ident(),
        gpu_peak_allocated_bytes=peak_allocated,
        gpu_peak_reserved_bytes=peak_reserved,
        torch_determinism_observed=determinism_after,
    )


def _warm_neural_query_process_slot(
    worker_cpu_threads: int,
) -> tuple[int, int, int]:
    """Configure and prove one spawned slot without initializing CUDA."""

    threads = int(worker_cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    return (
        os.getpid(),
        int(torch.get_num_threads()),
        int(torch.get_num_interop_threads()),
    )


class _LeaseExecutors:
    """One stable single-worker executor for every active device slot."""

    def __init__(
        self,
        resource_plan: RoleNeutralNeuralQueryTaskResourcePlan,
        *,
        parallelism: int,
    ) -> None:
        self.resource_plan = resource_plan
        self.parallelism = int(parallelism)
        self._executors: tuple[concurrent.futures.Executor, ...] = ()

    def __enter__(self) -> "_LeaseExecutors":
        if self._executors:
            raise RuntimeError(
                "neural-query lease executor cannot be entered twice"
            )
        if self.resource_plan.fold_parallel_backend == "processes":
            context = mp.get_context("spawn")
            self._executors = tuple(
                concurrent.futures.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=context,
                )
                for _slot in range(self.parallelism)
            )
            try:
                warm_futures = tuple(
                    executor.submit(
                        _warm_neural_query_process_slot,
                        self.resource_plan.worker_cpu_threads,
                    )
                    for executor in self._executors
                )
                worker_reports = tuple(
                    future.result() for future in warm_futures
                )
                worker_pids = tuple(
                    int(report[0]) for report in worker_reports
                )
                if len(set(worker_pids)) != len(worker_pids):
                    raise RuntimeError(
                        "neural-query process lease slots did not receive "
                        "isolated workers"
                    )
                if any(
                    report[1:] != (
                        self.resource_plan.worker_cpu_threads,
                        self.resource_plan.worker_cpu_threads,
                    )
                    for report in worker_reports
                ):
                    raise RuntimeError(
                        "neural-query process slot failed to bind its Torch "
                        "CPU thread lease"
                    )
            except BaseException:
                executors = self._executors
                self._executors = ()
                for executor in executors:
                    executor.shutdown(wait=True, cancel_futures=True)
                raise
        else:
            self._executors = tuple(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"oci-neural-query-{slot:02d}",
                )
                for slot in range(self.parallelism)
            )
        return self

    def submit(
        self,
        *,
        slot: int,
        worker: Callable[[Any, str], Any],
        task: Any,
        device: str,
    ) -> concurrent.futures.Future[CompletedNeuralQueryTask]:
        if not self._executors:
            raise RuntimeError("neural-query lease executor is not active")
        if slot < 0 or slot >= len(self._executors):
            raise ValueError("neural-query executor received an invalid slot")
        return self._executors[slot].submit(
            _invoke_neural_query_task,
            worker,
            task,
            device,
            worker_cpu_threads=self.resource_plan.worker_cpu_threads,
            process_isolated=(
                self.resource_plan.fold_parallel_backend == "processes"
            ),
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        executors = self._executors
        self._executors = ()
        for executor in executors:
            executor.shutdown(
                wait=True,
                cancel_futures=exc_type is not None,
            )


def _maximum_overlap(
    intervals: Sequence[Mapping[str, Any]],
) -> int:
    boundaries = [
        (int(row["started_monotonic_ns"]), 1)
        for row in intervals
    ] + [
        (int(row["finished_monotonic_ns"]), -1)
        for row in intervals
    ]
    active = 0
    maximum = 0
    for _timestamp, delta in sorted(
        boundaries,
        key=lambda value: (value[0], value[1]),
    ):
        active += delta
        if active < 0:
            raise RuntimeError(
                "neural-query telemetry released an idle task lease"
            )
        maximum = max(maximum, active)
    if active != 0:
        raise RuntimeError(
            "neural-query telemetry left one task lease active"
        )
    return maximum


def execute_bounded_neural_query_tasks(
    tasks: Sequence[Any],
    *,
    task_names: Sequence[str],
    resource_plan: RoleNeutralNeuralQueryTaskResourcePlan,
    worker: Callable[[Any, str], Any],
    parallelism: int,
    phase: str,
) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
    """Execute canonical tasks while leases follow completion, never waiting."""

    rows = tuple(tasks)
    names = tuple(str(value) for value in task_names)
    if not rows or len(rows) != len(names) or len(names) != len(set(names)):
        raise ValueError(
            "neural-query bounded tasks require unique canonical names"
        )
    if not isinstance(
        resource_plan,
        RoleNeutralNeuralQueryTaskResourcePlan,
    ):
        raise TypeError(
            "neural-query bounded tasks require a typed resource plan"
        )
    if not callable(worker):
        raise TypeError("neural-query bounded task worker must be callable")
    configured = int(parallelism)
    if configured < 1:
        raise ValueError("neural-query task parallelism must be positive")
    active_count = min(configured, len(rows))
    devices = resource_plan.devices_for_parallelism(configured)[:active_count]
    completed_by_index: dict[int, CompletedNeuralQueryTask] = {}
    active_futures: dict[
        concurrent.futures.Future[CompletedNeuralQueryTask],
        tuple[int, int],
    ] = {}
    next_index = 0

    with _LeaseExecutors(
        resource_plan,
        parallelism=active_count,
    ) as executor:

        def submit_to_slot(slot: int) -> None:
            nonlocal next_index
            if next_index >= len(rows):
                return
            index = next_index
            next_index += 1
            future = executor.submit(
                slot=slot,
                worker=worker,
                task=rows[index],
                device=devices[slot],
            )
            active_futures[future] = (index, slot)

        for slot in range(active_count):
            submit_to_slot(slot)
        while active_futures:
            done, _pending = concurrent.futures.wait(
                tuple(active_futures),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            released: list[int] = []
            for future in sorted(
                done,
                key=lambda value: active_futures[value][0],
            ):
                index, slot = active_futures.pop(future)
                completed_by_index[index] = future.result()
                released.append(slot)
            for slot in sorted(released):
                submit_to_slot(slot)

    if set(completed_by_index) != set(range(len(rows))):
        raise RuntimeError(
            "neural-query executor omitted a canonical task"
        )
    completed = tuple(
        completed_by_index[index] for index in range(len(rows))
    )
    intervals = tuple(
        {
            "task": names[index],
            "canonical_task_index": index,
            "device": record.device,
            "process_id": record.process_id,
            "thread_id": record.thread_id,
            "started_monotonic_ns": record.started_monotonic_ns,
            "finished_monotonic_ns": record.finished_monotonic_ns,
            "gpu_peak_allocated_bytes": (
                record.gpu_peak_allocated_bytes
            ),
            "gpu_peak_reserved_bytes": (
                record.gpu_peak_reserved_bytes
            ),
            "torch_determinism_observed": (
                None
                if record.torch_determinism_observed is None
                else copy.deepcopy(
                    dict(record.torch_determinism_observed)
                )
            ),
        }
        for index, record in enumerate(completed)
    )
    maximum = _maximum_overlap(intervals)
    if active_count > 1 and len(rows) > 1 and maximum < 2:
        raise RuntimeError(
            "configured parallel neural-query tasks did not overlap"
        )
    if (
        resource_plan.fold_parallel_backend == "processes"
        and active_count > 1
        and len({row["process_id"] for row in intervals}) < 2
    ):
        raise RuntimeError(
            "parallel neural-query tasks were not process isolated"
        )
    selected = set(resource_plan.devices)
    observed = {str(row["device"]) for row in intervals}
    if (
        len(rows) >= len(selected)
        and configured >= len(selected)
        and observed != selected
    ):
        raise RuntimeError(
            "neural-query tasks did not use every selected device"
        )
    per_device: dict[str, dict[str, Any]] = {}
    for device in resource_plan.devices:
        device_rows = tuple(
            row for row in intervals if row["device"] == device
        )
        maximum_device = (
            _maximum_overlap(device_rows) if device_rows else 0
        )
        if maximum_device > resource_plan.fold_slots_per_device:
            raise RuntimeError(
                "neural-query tasks exceeded per-device slots"
            )
        allocated_peaks = [
            int(row["gpu_peak_allocated_bytes"])
            for row in device_rows
            if row["gpu_peak_allocated_bytes"] is not None
        ]
        reserved_peaks = [
            int(row["gpu_peak_reserved_bytes"])
            for row in device_rows
            if row["gpu_peak_reserved_bytes"] is not None
        ]
        per_device[device] = {
            "task_count": len(device_rows),
            "maximum_concurrent_leases": maximum_device,
            "maximum_child_peak_allocated_bytes": (
                max(allocated_peaks) if allocated_peaks else None
            ),
            "maximum_child_peak_reserved_bytes": (
                max(reserved_peaks) if reserved_peaks else None
            ),
            "maximum_child_peak_allocator_charge_bytes": (
                max(
                    (
                        max(
                            int(row["gpu_peak_allocated_bytes"]),
                            int(row["gpu_peak_reserved_bytes"]),
                        )
                        for row in device_rows
                        if row["gpu_peak_allocated_bytes"] is not None
                        and row["gpu_peak_reserved_bytes"] is not None
                    ),
                    default=0,
                )
                or None
            ),
        }
    body = {
        "schema_version": NEURAL_QUERY_TASK_PHASE_ATTESTATION_SCHEMA,
        "phase": str(phase),
        "configured_parallelism": configured,
        "actual_task_count": len(rows),
        "maximum_concurrent_leases": maximum,
        "task_intervals": list(intervals),
        "per_device": per_device,
        "configured_total_parallelism_respected": maximum <= configured,
        "configured_per_device_slots_respected": True,
        "waiting_tasks_hold_no_lease": True,
        "canonical_result_order_restored": True,
        "process_isolated": (
            resource_plan.fold_parallel_backend == "processes"
        ),
        "worker_cpu_threads": resource_plan.worker_cpu_threads,
        "resource_locators_in_scientific_payload": False,
    }
    attestation = {**body, "content_sha256": _sha256_json(body)}
    return (
        tuple(record.value for record in completed),
        attestation,
    )


__all__ = [
    "NEURAL_QUERY_AUTHENTICATED_CACHE_REFERENCE_SCHEMA",
    "NEURAL_QUERY_TASK_PHASE_ATTESTATION_SCHEMA",
    "NeuralQueryAuthenticatedCacheReference",
    "execute_bounded_neural_query_tasks",
]
