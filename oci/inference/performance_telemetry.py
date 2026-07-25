"""Low-overhead subphase telemetry for portable workflow benchmarking."""

from __future__ import annotations

import math
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
    identity_sha256,
)

TELEMETRY_SCHEMA = "portable_workflow_subphase_telemetry_v1"
PERFORMANCE_ACCEPTANCE_POLICY_SCHEMA = "portable_workflow_performance_acceptance_policy_v1"
PERFORMANCE_BENCHMARK_RUN_SCHEMA = "portable_workflow_benchmark_run_v1"
PERFORMANCE_ACCEPTANCE_SCHEMA = "portable_workflow_performance_acceptance_v2"
PERFORMANCE_SCIENTIFIC_RESULT_SCHEMA = "portable_workflow_benchmark_scientific_result_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVITY_KINDS = frozenset(
    {
        "ordinary",
        "model_fit",
        "coordination_proof",
        "terminal_audit",
    }
)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite nonnegative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return normalized


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class RepresentativeScope:
    """One configured representative fit size; labels are opaque to the code."""

    label: str
    fit_row_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("representative scope label must be nonempty")
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(
            self,
            "fit_row_count",
            _positive_integer(
                self.fit_row_count,
                label=f"representative scope {self.label!r} fit_row_count",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerformanceAcceptancePolicy:
    """Operational benchmark gates supplied entirely by configuration."""

    representative_scopes: tuple[RepresentativeScope, ...]
    resource_performance_safety: ResourcePerformanceSafetyPolicy
    scientific_reference_candidate: str
    multi_device_baselines: tuple[tuple[str, str], ...]
    schema_version: str = PERFORMANCE_ACCEPTANCE_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_ACCEPTANCE_POLICY_SCHEMA:
            raise ValueError("unsupported performance acceptance policy schema")
        scopes = tuple(self.representative_scopes)
        if (
            not scopes
            or any(not isinstance(value, RepresentativeScope) for value in scopes)
            or len({value.label for value in scopes}) != len(scopes)
        ):
            raise ValueError("performance policy requires unique representative scopes")
        object.__setattr__(self, "representative_scopes", scopes)
        if not isinstance(
            self.resource_performance_safety,
            ResourcePerformanceSafetyPolicy,
        ):
            raise TypeError("performance acceptance requires typed " "resource_performance_safety")
        if (
            not isinstance(self.scientific_reference_candidate, str)
            or not self.scientific_reference_candidate.strip()
        ):
            raise ValueError("scientific_reference_candidate must be nonempty")
        object.__setattr__(
            self,
            "scientific_reference_candidate",
            self.scientific_reference_candidate.strip(),
        )
        baselines = tuple(
            (str(candidate).strip(), str(baseline).strip())
            for candidate, baseline in self.multi_device_baselines
        )
        if any(
            not candidate or not baseline or candidate == baseline
            for candidate, baseline in baselines
        ) or len({candidate for candidate, _baseline in baselines}) != len(baselines):
            raise ValueError("multi-device baseline bindings are invalid")
        object.__setattr__(self, "multi_device_baselines", baselines)

    @property
    def maximum_coordination_overhead_ratio(self) -> float:
        return self.resource_performance_safety.maximum_coordination_proof_overhead_ratio

    @property
    def maximum_ordinary_read_amplification(self) -> float:
        return self.resource_performance_safety.maximum_ordinary_read_amplification

    @property
    def maximum_peak_allocation_fraction(self) -> float:
        return self.resource_performance_safety.gpu_max_allocation_fraction

    @property
    def minimum_headroom_bytes(self) -> int:
        return self.resource_performance_safety.gpu_minimum_headroom_bytes

    @property
    def minimum_repetitions_per_scope(self) -> int:
        return self.resource_performance_safety.minimum_benchmark_repetitions_per_scope

    @property
    def minimum_multi_device_speedup(self) -> float:
        return self.resource_performance_safety.minimum_multi_device_throughput_ratio

    @property
    def read_counter_source(self) -> str:
        return self.resource_performance_safety.read_counter_source

    @property
    def scope_by_label(self) -> Mapping[str, RepresentativeScope]:
        return {value.label: value for value in self.representative_scopes}

    @property
    def baseline_by_candidate(self) -> Mapping[str, str]:
        return dict(self.multi_device_baselines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "representative_scopes": [value.as_dict() for value in self.representative_scopes],
            "resource_performance_safety": (self.resource_performance_safety.as_dict()),
            "scientific_reference_candidate": (self.scientific_reference_candidate),
            "multi_device_baselines": [
                {"candidate": candidate, "baseline": baseline}
                for candidate, baseline in self.multi_device_baselines
            ],
        }


@dataclass(frozen=True)
class ImmutableInputObservation:
    """Path-neutral identity and size of one immutable workflow input."""

    content_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(str(self.content_sha256)) is None:
            raise ValueError("immutable input content_sha256 must be SHA-256")
        object.__setattr__(
            self,
            "size_bytes",
            _positive_integer(
                self.size_bytes,
                label="immutable input size_bytes",
            ),
        )


@dataclass(frozen=True)
class BenchmarkRunObservation:
    """One measured candidate/scope repetition with raw safety evidence."""

    candidate_name: str
    scope_label: str
    repetition_index: int
    device_ids: tuple[str, ...]
    concurrency_per_device: int
    completed_scope_fits: int
    model_fit_wall_seconds: float
    peak_allocation_fraction: float | None
    minimum_observed_headroom_bytes: int | None
    oom_observed: bool
    scientific_artifact_sha256: str | None
    artifact_path: str | None
    end_to_end_wall_seconds: float | None = None
    complete_artifacts_exactly_equal: bool = True
    schema_version: str = PERFORMANCE_BENCHMARK_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_BENCHMARK_RUN_SCHEMA:
            raise ValueError("unsupported benchmark-run schema")
        for name in ("candidate_name", "scope_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"benchmark run {name} must be nonempty")
            object.__setattr__(self, name, value.strip())
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or self.repetition_index < 0
        ):
            raise ValueError("benchmark repetition_index must be nonnegative")
        devices = tuple(str(value).strip() for value in self.device_ids)
        if (
            not devices
            or any(not value for value in devices)
            or len(devices) != len(set(devices))
            or ("cpu" in devices and devices != ("cpu",))
        ):
            raise ValueError("benchmark device_ids are invalid")
        object.__setattr__(self, "device_ids", devices)
        object.__setattr__(
            self,
            "concurrency_per_device",
            _positive_integer(
                self.concurrency_per_device,
                label="benchmark concurrency_per_device",
            ),
        )
        if (
            isinstance(self.completed_scope_fits, bool)
            or not isinstance(self.completed_scope_fits, int)
            or self.completed_scope_fits < 0
            or (not self.oom_observed and self.completed_scope_fits < 1)
        ):
            raise ValueError("benchmark completed_scope_fits are invalid")
        wall = _finite_nonnegative(
            self.model_fit_wall_seconds,
            label="benchmark model_fit_wall_seconds",
        )
        if wall <= 0:
            raise ValueError("benchmark model_fit_wall_seconds must be positive")
        object.__setattr__(self, "model_fit_wall_seconds", wall)
        if self.end_to_end_wall_seconds is not None:
            end_to_end = _finite_nonnegative(
                self.end_to_end_wall_seconds,
                label="benchmark end_to_end_wall_seconds",
            )
            if end_to_end <= 0:
                raise ValueError(
                    "benchmark end_to_end_wall_seconds must be positive"
                )
            object.__setattr__(
                self,
                "end_to_end_wall_seconds",
                end_to_end,
            )
        if not isinstance(self.complete_artifacts_exactly_equal, bool):
            raise TypeError(
                "benchmark complete_artifacts_exactly_equal must be boolean"
            )
        if not isinstance(self.oom_observed, bool):
            raise TypeError("benchmark oom_observed must be boolean")
        if devices == ("cpu",):
            if (
                self.peak_allocation_fraction is not None
                or self.minimum_observed_headroom_bytes is not None
            ):
                raise ValueError("CPU benchmark runs cannot report accelerator memory")
        else:
            peak = _finite_nonnegative(
                self.peak_allocation_fraction,
                label="benchmark peak_allocation_fraction",
            )
            if peak > 1:
                raise ValueError("benchmark peak_allocation_fraction cannot exceed one")
            object.__setattr__(self, "peak_allocation_fraction", peak)
            if (
                isinstance(self.minimum_observed_headroom_bytes, bool)
                or not isinstance(self.minimum_observed_headroom_bytes, int)
                or self.minimum_observed_headroom_bytes < 0
            ):
                raise ValueError("benchmark minimum_observed_headroom_bytes is invalid")
        if self.oom_observed:
            if self.scientific_artifact_sha256 is not None:
                raise ValueError("OOM runs cannot claim a scientific artifact")
        elif _SHA256.fullmatch(str(self.scientific_artifact_sha256)) is None:
            raise ValueError("completed benchmark runs require a scientific artifact SHA-256")
        if self.artifact_path is not None and not isinstance(
            self.artifact_path,
            str,
        ):
            raise TypeError("benchmark artifact_path must be text or null")

    @property
    def accelerator_count(self) -> int:
        return 0 if self.device_ids == ("cpu",) else len(self.device_ids)

    @property
    def measured_wall_seconds(self) -> float:
        """End-to-end candidate wall time, with a legacy fit-only fallback."""

        if self.end_to_end_wall_seconds is not None:
            return float(self.end_to_end_wall_seconds)
        return float(self.model_fit_wall_seconds)


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
    requested = set(indices)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            index = int(parts[0])
            if index not in requested:
                continue
            rows.append(
                {
                    "device": f"cuda:{index}",
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


def assess_performance_acceptance(
    telemetry: Mapping[str, Any],
    *,
    model_fit_subphases: Sequence[str],
    coordination_subphases: Sequence[str],
    terminal_audit_subphases: Sequence[str],
    unique_immutable_input_bytes: int,
    overhead_limit: float,
    read_amplification_limit: float,
) -> Mapping[str, Any]:
    """Compatibility evaluator with all names and thresholds caller-supplied."""

    records = telemetry.get("subphases")
    if not isinstance(records, list):
        raise ValueError("telemetry lacks subphase records")
    fit_names = set(model_fit_subphases)
    coordination_names = set(coordination_subphases)
    terminal_names = set(terminal_audit_subphases)
    if (
        not fit_names
        or not coordination_names
        or len(terminal_names) != 1
        or fit_names & coordination_names
        or fit_names & terminal_names
        or coordination_names & terminal_names
    ):
        raise ValueError(
            "performance subphase roles must be nonempty and disjoint, with "
            "exactly one terminal-audit name"
        )
    overhead_limit = _finite_nonnegative(
        overhead_limit,
        label="overhead_limit",
    )
    read_amplification_limit = _finite_nonnegative(
        read_amplification_limit,
        label="read_amplification_limit",
    )
    if overhead_limit <= 0 or read_amplification_limit <= 0:
        raise ValueError("performance acceptance ratios must be positive")
    if (
        isinstance(unique_immutable_input_bytes, bool)
        or not isinstance(unique_immutable_input_bytes, int)
        or unique_immutable_input_bytes < 1
    ):
        raise ValueError("unique_immutable_input_bytes must be positive")
    fit_wall = sum(
        float(row["wall_seconds"])
        for row in records
        if row.get("name") in fit_names and row.get("status") == "completed"
    )
    coordination_wall = sum(
        float(row["wall_seconds"])
        for row in records
        if row.get("name") in coordination_names and row.get("status") == "completed"
    )
    ordinary_reads = sum(
        int(row.get("process_read_bytes") or 0)
        for row in records
        if row.get("name") not in terminal_names
    )
    terminal_reads = sum(
        int(row.get("process_read_bytes") or 0)
        for row in records
        if row.get("name") in terminal_names
    )
    observed_terminal = [row for row in records if row.get("name") in terminal_names]
    overhead_ratio = float("inf") if fit_wall <= 0 else coordination_wall / fit_wall
    read_ratio = ordinary_reads / int(unique_immutable_input_bytes)
    terminal_audit_proved = (
        len(observed_terminal) == 1 and observed_terminal[0].get("status") == "completed"
    )
    return {
        "schema_version": "portable_workflow_performance_acceptance_v1",
        "model_fit_wall_seconds": fit_wall,
        "coordination_proof_wall_seconds": coordination_wall,
        "coordination_overhead_ratio": overhead_ratio,
        "coordination_overhead_limit": float(overhead_limit),
        "ordinary_process_read_bytes": ordinary_reads,
        "terminal_audit_process_read_bytes": terminal_reads,
        "terminal_audit_subphase": next(iter(terminal_names)),
        "exactly_one_completed_terminal_audit": terminal_audit_proved,
        "unique_immutable_input_bytes": int(unique_immutable_input_bytes),
        "ordinary_read_amplification": read_ratio,
        "ordinary_read_amplification_limit": float(read_amplification_limit),
        "coordination_target_accepted": overhead_ratio <= float(overhead_limit),
        "read_target_accepted": read_ratio <= float(read_amplification_limit),
        "accepted": (
            overhead_ratio <= float(overhead_limit)
            and read_ratio <= float(read_amplification_limit)
            and terminal_audit_proved
        ),
    }


def _record_read_bytes(
    record: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> int:
    if source == "process_read_bytes":
        value = record.get("process_read_bytes")
    else:
        counters = record.get("byte_counters")
        value = counters.get("read") if isinstance(counters, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"telemetry record {index} lacks valid {source} evidence")
    return int(value)


def _unique_immutable_input_summary(
    observations: Sequence[ImmutableInputObservation],
) -> tuple[list[dict[str, Any]], int]:
    identities: dict[str, int] = {}
    for value in observations:
        if not isinstance(value, ImmutableInputObservation):
            raise TypeError("immutable input observations must use the typed contract")
        existing = identities.get(value.content_sha256)
        if existing is not None and existing != value.size_bytes:
            raise ValueError("one immutable content identity has conflicting byte sizes")
        identities[value.content_sha256] = value.size_bytes
    if not identities:
        raise ValueError("at least one immutable input observation is required")
    rows = [
        {
            "content_sha256": digest,
            "size_bytes": identities[digest],
        }
        for digest in sorted(identities)
    ]
    return rows, sum(identities.values())


def _assess_tagged_telemetry(
    telemetry: Mapping[str, Any],
    *,
    policy: PerformanceAcceptancePolicy,
    unique_input_bytes: int,
) -> dict[str, Any]:
    records = telemetry.get("subphases")
    if not isinstance(records, list):
        raise ValueError("telemetry lacks subphase records")
    scope_labels = set(policy.scope_by_label)
    fit_wall = 0.0
    coordination_wall = 0.0
    ordinary_reads = 0
    terminal_reads = 0
    terminal_records: list[Mapping[str, Any]] = []
    failed_representative_records: list[str] = []
    fit_counts = {label: 0 for label in scope_labels}
    coordination_counts = {label: 0 for label in scope_labels}
    scope_wall = {
        label: {"model_fit_wall_seconds": 0.0, "coordination_wall_seconds": 0.0}
        for label in scope_labels
    }
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"telemetry record {index} is not an object")
        kind = raw.get("activity_kind", "ordinary")
        if kind not in _ACTIVITY_KINDS:
            raise ValueError(f"telemetry record {index} has an invalid activity_kind")
        status = raw.get("status")
        if status not in {"completed", "failed"}:
            raise ValueError(f"telemetry record {index} has an invalid status")
        wall = _finite_nonnegative(
            raw.get("wall_seconds"),
            label=f"telemetry record {index} wall_seconds",
        )
        read_bytes = _record_read_bytes(
            raw,
            source=policy.read_counter_source,
            index=index,
        )
        if kind == "terminal_audit":
            terminal_records.append(raw)
            terminal_reads += read_bytes
        else:
            ordinary_reads += read_bytes
        label = raw.get("scope_label")
        if kind in {"model_fit", "coordination_proof"} and label in scope_labels:
            if status != "completed":
                failed_representative_records.append(str(raw.get("name")))
            if kind == "model_fit":
                fit_wall += wall
                fit_counts[str(label)] += 1
                scope_wall[str(label)]["model_fit_wall_seconds"] += wall
            else:
                coordination_wall += wall
                coordination_counts[str(label)] += 1
                scope_wall[str(label)]["coordination_wall_seconds"] += wall
    missing_fit_scopes = sorted(label for label, count in fit_counts.items() if count < 1)
    missing_coordination_scopes = sorted(
        label for label, count in coordination_counts.items() if count < 1
    )
    terminal_audit_accepted = (
        len(terminal_records) == 1 and terminal_records[0].get("status") == "completed"
    )
    overhead_ratio = float("inf") if fit_wall <= 0 else coordination_wall / fit_wall
    read_ratio = float("inf") if unique_input_bytes <= 0 else ordinary_reads / unique_input_bytes
    evidence_complete = (
        not missing_fit_scopes
        and not missing_coordination_scopes
        and not failed_representative_records
    )
    scope_rows = []
    for scope in policy.representative_scopes:
        values = scope_wall[scope.label]
        fit_value = values["model_fit_wall_seconds"]
        scope_rows.append(
            {
                "scope_label": scope.label,
                "configured_fit_row_count": scope.fit_row_count,
                **values,
                "coordination_overhead_ratio": (
                    float("inf")
                    if fit_value <= 0
                    else values["coordination_wall_seconds"] / fit_value
                ),
            }
        )
    coordination_accepted = (
        evidence_complete and overhead_ratio <= policy.maximum_coordination_overhead_ratio
    )
    read_accepted = (
        terminal_audit_accepted and read_ratio <= policy.maximum_ordinary_read_amplification
    )
    return {
        "representative_scope_telemetry": scope_rows,
        "model_fit_wall_seconds": fit_wall,
        "coordination_proof_wall_seconds": coordination_wall,
        "coordination_overhead_ratio": overhead_ratio,
        "coordination_overhead_limit": (policy.maximum_coordination_overhead_ratio),
        "ordinary_read_bytes": ordinary_reads,
        "terminal_audit_read_bytes": terminal_reads,
        "unique_immutable_input_bytes": unique_input_bytes,
        "ordinary_read_amplification": read_ratio,
        "ordinary_read_amplification_limit": (policy.maximum_ordinary_read_amplification),
        "terminal_audit_record_count": len(terminal_records),
        "exactly_one_completed_terminal_audit": terminal_audit_accepted,
        "missing_model_fit_scopes": missing_fit_scopes,
        "missing_coordination_scopes": missing_coordination_scopes,
        "failed_representative_records": failed_representative_records,
        "representative_telemetry_complete": evidence_complete,
        "coordination_target_accepted": coordination_accepted,
        "read_target_accepted": read_accepted,
        "accepted": coordination_accepted and read_accepted,
    }


def _candidate_benchmark_summaries(
    *,
    policy: PerformanceAcceptancePolicy,
    benchmark_runs: Sequence[BenchmarkRunObservation],
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    if not benchmark_runs:
        return [], None, None
    scope_by_label = policy.scope_by_label
    grouped: dict[str, list[BenchmarkRunObservation]] = {}
    for value in benchmark_runs:
        if not isinstance(value, BenchmarkRunObservation):
            raise TypeError("benchmark runs must use the typed observation contract")
        if value.scope_label not in scope_by_label:
            raise ValueError(f"benchmark run names an unconfigured scope: {value.scope_label}")
        grouped.setdefault(value.candidate_name, []).append(value)

    preliminary: dict[str, dict[str, Any]] = {}
    for candidate_name in sorted(grouped):
        runs = grouped[candidate_name]
        accelerator_counts = {value.accelerator_count for value in runs}
        execution_device_counts = {len(value.device_ids) for value in runs}
        concurrencies = {value.concurrency_per_device for value in runs}
        configuration_consistent = (
            len(accelerator_counts) == 1
            and len(execution_device_counts) == 1
            and len(concurrencies) == 1
        )
        accelerator_count = next(iter(accelerator_counts)) if len(accelerator_counts) == 1 else None
        execution_device_count = (
            next(iter(execution_device_counts)) if len(execution_device_counts) == 1 else None
        )
        concurrency = next(iter(concurrencies)) if len(concurrencies) == 1 else None
        scope_rows: list[dict[str, Any]] = []
        candidate_fit_rows = 0
        candidate_wall = 0.0
        for scope in policy.representative_scopes:
            selected_runs = [value for value in runs if value.scope_label == scope.label]
            repetition_indices = [value.repetition_index for value in selected_runs]
            unique_repetitions = len(repetition_indices) == len(set(repetition_indices))
            repetition_accepted = (
                unique_repetitions and len(selected_runs) >= policy.minimum_repetitions_per_scope
            )
            oom_count = sum(value.oom_observed for value in selected_runs)
            completed = [value for value in selected_runs if not value.oom_observed]
            hashes = {
                str(value.scientific_artifact_sha256)
                for value in completed
                if value.scientific_artifact_sha256 is not None
            }
            deterministic = (
                bool(selected_runs)
                and len(completed) == len(selected_runs)
                and len(hashes) == 1
                and all(
                    value.complete_artifacts_exactly_equal
                    for value in selected_runs
                )
            )
            memory_accepted = bool(selected_runs) and all(
                value.accelerator_count == 0
                or (
                    value.peak_allocation_fraction is not None
                    and value.peak_allocation_fraction < policy.maximum_peak_allocation_fraction
                    and value.minimum_observed_headroom_bytes is not None
                    and value.minimum_observed_headroom_bytes >= policy.minimum_headroom_bytes
                )
                for value in selected_runs
            )
            fit_rows = sum(value.completed_scope_fits * scope.fit_row_count for value in completed)
            wall = sum(value.measured_wall_seconds for value in selected_runs)
            candidate_fit_rows += fit_rows
            candidate_wall += wall
            scope_rows.append(
                {
                    "scope_label": scope.label,
                    "configured_fit_row_count": scope.fit_row_count,
                    "run_count": len(selected_runs),
                    "repetition_indices_unique": unique_repetitions,
                    "minimum_repetitions_accepted": repetition_accepted,
                    "oom_count": oom_count,
                    "zero_oom_accepted": oom_count == 0 and bool(selected_runs),
                    "deterministic_artifact_identity": deterministic,
                    "scientific_artifact_sha256": (next(iter(hashes)) if deterministic else None),
                    "peak_allocation_and_headroom_accepted": memory_accepted,
                    "completed_fit_rows": fit_rows,
                    "measured_wall_seconds": wall,
                    "throughput_fit_rows_per_second": (0.0 if wall <= 0 else fit_rows / wall),
                }
            )
        preliminary_accepted = configuration_consistent and all(
            row["minimum_repetitions_accepted"]
            and row["zero_oom_accepted"]
            and row["deterministic_artifact_identity"]
            and row["peak_allocation_and_headroom_accepted"]
            for row in scope_rows
        )
        preliminary[candidate_name] = {
            "candidate_name": candidate_name,
            "accelerator_count": accelerator_count,
            "execution_device_count": execution_device_count,
            "concurrency_per_device": concurrency,
            "configuration_consistent_across_runs": configuration_consistent,
            "device_assignments_observed": sorted({tuple(value.device_ids) for value in runs}),
            "artifact_paths_observed": sorted(
                {value.artifact_path for value in runs if value.artifact_path is not None}
            ),
            "scope_results": scope_rows,
            "completed_fit_rows": candidate_fit_rows,
            "measured_wall_seconds": candidate_wall,
            "throughput_fit_rows_per_second": (
                0.0 if candidate_wall <= 0 else candidate_fit_rows / candidate_wall
            ),
            "preliminary_safety_and_repeatability_accepted": (preliminary_accepted),
        }

    reference = preliminary.get(policy.scientific_reference_candidate)
    reference_hashes: dict[str, str] = {}
    reference_accepted = bool(
        reference and reference["preliminary_safety_and_repeatability_accepted"]
    )
    if reference is not None:
        reference_hashes = {
            str(row["scope_label"]): str(row["scientific_artifact_sha256"])
            for row in reference["scope_results"]
            if row["scientific_artifact_sha256"] is not None
        }
    baselines = policy.baseline_by_candidate
    summaries: list[dict[str, Any]] = []
    for candidate_name in sorted(preliminary):
        row = preliminary[candidate_name]
        scientific_equal = (
            reference_accepted
            and len(reference_hashes) == len(policy.representative_scopes)
            and all(
                value["scientific_artifact_sha256"]
                == reference_hashes.get(str(value["scope_label"]))
                for value in row["scope_results"]
            )
        )
        accelerator_count = row["accelerator_count"]
        speedup_rows: list[dict[str, Any]] = []
        if accelerator_count is not None and accelerator_count > 1:
            baseline_name = baselines.get(candidate_name)
            baseline = preliminary.get(str(baseline_name))
            baseline_valid = bool(
                baseline_name
                and baseline
                and baseline["accelerator_count"] == 1
                and baseline["preliminary_safety_and_repeatability_accepted"]
            )
            baseline_scope_rows = (
                {str(value["scope_label"]): value for value in baseline["scope_results"]}
                if baseline_valid and baseline is not None
                else {}
            )
            for value in row["scope_results"]:
                baseline_scope = baseline_scope_rows.get(str(value["scope_label"]))
                baseline_throughput = (
                    None
                    if baseline_scope is None
                    else float(baseline_scope["throughput_fit_rows_per_second"])
                )
                speedup = (
                    0.0
                    if not baseline_throughput
                    else float(value["throughput_fit_rows_per_second"]) / baseline_throughput
                )
                speedup_rows.append(
                    {
                        "scope_label": value["scope_label"],
                        "baseline_candidate_name": baseline_name,
                        "speedup": speedup,
                        "minimum_required_speedup": (policy.minimum_multi_device_speedup),
                        "accepted": (
                            baseline_valid and speedup >= policy.minimum_multi_device_speedup
                        ),
                    }
                )
            multi_device_claim_accepted = bool(speedup_rows) and all(
                value["accepted"] for value in speedup_rows
            )
        else:
            multi_device_claim_accepted = True
        accepted = (
            row["preliminary_safety_and_repeatability_accepted"]
            and scientific_equal
            and multi_device_claim_accepted
        )
        summaries.append(
            {
                **row,
                "scientifically_equal_to_reference": scientific_equal,
                "scientific_reference_candidate": (policy.scientific_reference_candidate),
                "multi_device_speedup_results": speedup_rows,
                "multi_device_throughput_claim_accepted": (multi_device_claim_accepted),
                "accepted": accepted,
            }
        )

    accepted = [value for value in summaries if value["accepted"]]
    selected_candidate = (
        None
        if not accepted
        else min(
            accepted,
            key=lambda value: (
                -float(value["throughput_fit_rows_per_second"]),
                int(value["execution_device_count"]) * int(value["concurrency_per_device"]),
                int(value["execution_device_count"]),
                str(value["candidate_name"]),
            ),
        )["candidate_name"]
    )
    scientific_identity = None
    if reference_accepted and len(reference_hashes) == len(policy.representative_scopes):
        scientific_body = {
            "schema_version": PERFORMANCE_SCIENTIFIC_RESULT_SCHEMA,
            "scope_results": [
                {
                    "scope_label": scope.label,
                    "fit_row_count": scope.fit_row_count,
                    "scientific_artifact_sha256": reference_hashes[scope.label],
                }
                for scope in policy.representative_scopes
            ],
        }
        scientific_identity = identity_sha256(scientific_body)
    return summaries, selected_candidate, scientific_identity


def assess_benchmark_acceptance(
    telemetry: Mapping[str, Any],
    *,
    policy: PerformanceAcceptancePolicy,
    immutable_inputs: Sequence[ImmutableInputObservation],
    benchmark_runs: Sequence[BenchmarkRunObservation],
) -> Mapping[str, Any]:
    """Evaluate telemetry, repeatability, memory, and throughput evidence.

    Representative labels and row counts are opaque configured values; the
    evaluator never recognizes a particular cohort size.  Paths, device IDs,
    timing, and policy thresholds remain operational and are excluded from the
    returned scientific-result identity.
    """

    if not isinstance(policy, PerformanceAcceptancePolicy):
        raise TypeError("performance acceptance requires a typed policy")
    input_rows, unique_input_bytes = _unique_immutable_input_summary(immutable_inputs)
    telemetry_result = _assess_tagged_telemetry(
        telemetry,
        policy=policy,
        unique_input_bytes=unique_input_bytes,
    )
    candidate_results, selected_candidate, scientific_identity = _candidate_benchmark_summaries(
        policy=policy,
        benchmark_runs=benchmark_runs,
    )
    benchmark_accepted = selected_candidate is not None
    body = {
        "schema_version": PERFORMANCE_ACCEPTANCE_SCHEMA,
        "operational_policy": policy.as_dict(),
        "operational_policy_sha256": identity_sha256(policy.as_dict()),
        "immutable_inputs": input_rows,
        "telemetry_acceptance": telemetry_result,
        "candidate_results": candidate_results,
        "selected_candidate": selected_candidate,
        "benchmark_candidate_accepted": benchmark_accepted,
        "scientific_result_identity_sha256": scientific_identity,
        "scientific_identity_excludes_paths_devices_and_performance": True,
        "accepted": telemetry_result["accepted"] and benchmark_accepted,
    }
    return body


__all__ = [
    "BenchmarkRunObservation",
    "ByteCounters",
    "ImmutableInputObservation",
    "PERFORMANCE_ACCEPTANCE_POLICY_SCHEMA",
    "PERFORMANCE_ACCEPTANCE_SCHEMA",
    "PERFORMANCE_BENCHMARK_RUN_SCHEMA",
    "PerformanceAcceptancePolicy",
    "RepresentativeScope",
    "SubphaseTelemetry",
    "TELEMETRY_SCHEMA",
    "TelemetryLedger",
    "assess_benchmark_acceptance",
    "assess_performance_acceptance",
    "sample_nvidia_gpus",
]
