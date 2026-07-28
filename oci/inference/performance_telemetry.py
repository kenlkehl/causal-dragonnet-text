"""Operational telemetry for the science-first production workflow."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .portable_resource_scheduler import (
    _logical_to_physical_cuda_indices,
)

TELEMETRY_SCHEMA = "portable_workflow_subphase_telemetry_v1"
_ACTIVITY_KINDS = frozenset(
    {
        "ordinary",
        "model_fit",
        "coordination_proof",
        "terminal_audit",
    }
)


@dataclass
class ByteCounters:
    read: int = 0
    written: int = 0
    copied: int = 0
    hashed: int = 0
    compressed: int = 0
    decompressed: int = 0
    json_encoded: int = 0
    json_decoded: int = 0
    fsynced: int = 0

    def add(self, **values: int) -> None:
        for name, value in values.items():
            if name not in self.__dataclass_fields__:
                raise ValueError(f"unknown byte counter {name!r}")
            if int(value) < 0:
                raise ValueError("byte counters cannot decrease")
            setattr(self, name, int(getattr(self, name)) + int(value))

    def delta(self, before: "ByteCounters") -> "ByteCounters":
        return ByteCounters(
            **{
                name: int(getattr(self, name)) - int(getattr(before, name))
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class SubphaseTelemetry:
    name: str
    wall_seconds: float
    cpu_seconds: float
    process_read_bytes: int | None
    process_written_bytes: int | None
    byte_counters: Mapping[str, int]
    gpu_samples: tuple[Mapping[str, Any], ...]
    gpu_peak_memory_bytes: Mapping[str, int]
    status: str
    activity_kind: str = "ordinary"
    scope_label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "byte_counters": dict(self.byte_counters),
            "gpu_samples": [dict(value) for value in self.gpu_samples],
            "gpu_peak_memory_bytes": dict(self.gpu_peak_memory_bytes),
        }


def _proc_io() -> tuple[int, int] | None:
    path = Path("/proc/self/io")
    try:
        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name.strip()] = int(raw.strip())
        return int(values["read_bytes"]), int(values["write_bytes"])
    except (OSError, KeyError, ValueError):
        return None


def sample_nvidia_gpus(devices: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    indices = [
        int(value.split(":", 1)[1])
        for value in devices
        if str(value).startswith("cuda:") and str(value).split(":", 1)[1].isdigit()
    ]
    if not indices:
        return ()
    logical_to_physical = _logical_to_physical_cuda_indices(indices)
    physical_to_logical = {
        physical: logical
        for logical, physical in logical_to_physical.items()
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    requested = set(physical_to_logical)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            physical_index = int(parts[0])
            if physical_index not in requested:
                continue
            logical_index = physical_to_logical[physical_index]
            rows.append(
                {
                    "device": f"cuda:{logical_index}",
                    "uuid": parts[1],
                    "utilization_percent": float(parts[2]),
                    "memory_used_bytes": int(parts[3]) * 1024 * 1024,
                    "memory_total_bytes": int(parts[4]) * 1024 * 1024,
                }
            )
        except ValueError:
            continue
    return tuple(sorted(rows, key=lambda row: row["device"]))


def _reset_torch_peaks(devices: Sequence[str]) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for value in devices:
            if str(value).startswith("cuda:"):
                torch.cuda.reset_peak_memory_stats(int(str(value).split(":", 1)[1]))
    except (ImportError, RuntimeError):
        return


def _torch_peaks(devices: Sequence[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return values
        for value in devices:
            if str(value).startswith("cuda:"):
                index = int(str(value).split(":", 1)[1])
                values[str(value)] = max(
                    int(torch.cuda.max_memory_allocated(index)),
                    int(torch.cuda.max_memory_reserved(index)),
                )
    except (ImportError, RuntimeError):
        return {}
    return values


class TelemetryLedger:
    """Thread-safe counters plus explicit real-subphase timing spans."""

    def __init__(self, *, devices: Sequence[str] = ()) -> None:
        self.devices = tuple(str(value) for value in devices)
        self._bytes = ByteCounters()
        self._records: list[SubphaseTelemetry] = []
        self._lock = threading.RLock()

    @property
    def byte_counters(self) -> ByteCounters:
        with self._lock:
            return ByteCounters(**asdict(self._bytes))

    @property
    def records(self) -> tuple[SubphaseTelemetry, ...]:
        with self._lock:
            return tuple(self._records)

    def count_bytes(self, **values: int) -> None:
        with self._lock:
            self._bytes.add(**values)

    @contextmanager
    def subphase(
        self,
        name: str,
        *,
        activity_kind: str = "ordinary",
        scope_label: str | None = None,
    ) -> Iterator["TelemetryLedger"]:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("telemetry subphase name is required")
        if activity_kind not in _ACTIVITY_KINDS:
            raise ValueError("telemetry activity_kind is unsupported")
        if scope_label is not None and (
            not isinstance(scope_label, str) or not scope_label.strip()
        ):
            raise ValueError("telemetry scope_label must be nonempty or null")
        normalized_scope = None if scope_label is None else scope_label.strip()
        if activity_kind in {"model_fit", "coordination_proof"} and (normalized_scope is None):
            raise ValueError("fit and coordination telemetry require an explicit scope label")
        if activity_kind == "terminal_audit" and normalized_scope is not None:
            raise ValueError("terminal-audit telemetry cannot name one fit scope")
        with self._lock:
            before_counters = ByteCounters(**asdict(self._bytes))
        before_io = _proc_io()
        before_gpu = sample_nvidia_gpus(self.devices)
        _reset_torch_peaks(self.devices)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        status = "completed"
        try:
            yield self
        except BaseException:
            status = "failed"
            raise
        finally:
            wall = time.perf_counter() - wall_start
            cpu = time.process_time() - cpu_start
            after_io = _proc_io()
            after_gpu = sample_nvidia_gpus(self.devices)
            with self._lock:
                delta = self._bytes.delta(before_counters)
                self._records.append(
                    SubphaseTelemetry(
                        name=name,
                        wall_seconds=wall,
                        cpu_seconds=cpu,
                        process_read_bytes=(
                            None
                            if before_io is None or after_io is None
                            else max(0, after_io[0] - before_io[0])
                        ),
                        process_written_bytes=(
                            None
                            if before_io is None or after_io is None
                            else max(0, after_io[1] - before_io[1])
                        ),
                        byte_counters=asdict(delta),
                        gpu_samples=tuple((*before_gpu, *after_gpu)),
                        gpu_peak_memory_bytes=_torch_peaks(self.devices),
                        status=status,
                        activity_kind=activity_kind,
                        scope_label=normalized_scope,
                    )
                )

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            body = {
                "schema_version": TELEMETRY_SCHEMA,
                "devices": list(self.devices),
                "byte_counters": asdict(self._bytes),
                "subphases": [record.as_dict() for record in self._records],
            }
        return body


__all__ = [
    "ByteCounters",
    "SubphaseTelemetry",
    "TELEMETRY_SCHEMA",
    "TelemetryLedger",
    "sample_nvidia_gpus",
]
