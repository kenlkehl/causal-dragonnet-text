"""Descriptive CPU/GPU lane-overlap analysis for completed fit observations.

This module consumes directly measured monotonic intervals.  It never
reconstructs timestamps from durations or completion order and never treats
observed overlap as a causal speedup estimate.  Hardware locators, owner
identifiers, interval counts, and timestamp ranges are supplied entirely by
the completed observation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


FIT_LANE_INTERVAL_SCHEMA = "role_neutral_fit_lane_interval_v1"
COMPLETED_FIT_INTERVAL_OBSERVATION_SCHEMA = (
    "role_neutral_completed_fit_interval_observation_v1"
)
CPU_GPU_LANE_OVERLAP_ANALYSIS_SCHEMA = (
    "role_neutral_cpu_gpu_lane_overlap_analysis_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LANES = frozenset({"cpu", "gpu"})


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


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def _timestamp(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer nanosecond timestamp")
    return int(value)


def _fraction(numerator: int, denominator: int) -> float:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise RuntimeError("lane-overlap fraction inputs are inconsistent")
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True)
class FitLaneInterval:
    """One directly measured half-open lane interval, ``[start, finish)``."""

    interval_id: str
    owner_execution_id: str
    lane_kind: str
    subphase_name: str
    resource_id: str
    clock_domain_id: str
    started_monotonic_ns: int
    finished_monotonic_ns: int
    status: str
    timestamps_measured_directly: bool
    schema_version: str = FIT_LANE_INTERVAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FIT_LANE_INTERVAL_SCHEMA:
            raise ValueError("unsupported fit lane interval schema")
        for name in (
            "interval_id",
            "owner_execution_id",
            "subphase_name",
            "resource_id",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_text(getattr(self, name), label=name),
            )
        if self.lane_kind not in _LANES:
            raise ValueError("fit lane interval kind must be cpu or gpu")
        start = _timestamp(
            self.started_monotonic_ns,
            label="started_monotonic_ns",
        )
        finish = _timestamp(
            self.finished_monotonic_ns,
            label="finished_monotonic_ns",
        )
        if finish <= start:
            raise ValueError("fit lane interval must have positive duration")
        object.__setattr__(self, "started_monotonic_ns", start)
        object.__setattr__(self, "finished_monotonic_ns", finish)
        if self.status != "completed":
            raise ValueError("fit lane interval must be completed")
        if self.timestamps_measured_directly is not True:
            raise ValueError(
                "fit lane timestamps must be directly measured, not reconstructed"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FitLaneInterval":
        if not isinstance(value, Mapping):
            raise TypeError("fit lane interval must be one mapping")
        required = {
            "schema_version",
            "interval_id",
            "owner_execution_id",
            "lane_kind",
            "subphase_name",
            "resource_id",
            "clock_domain_id",
            "started_monotonic_ns",
            "finished_monotonic_ns",
            "status",
            "timestamps_measured_directly",
        }
        if set(value) != required:
            raise ValueError(
                "fit lane interval fields must be closed; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**copy.deepcopy(dict(value)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interval_id": self.interval_id,
            "owner_execution_id": self.owner_execution_id,
            "lane_kind": self.lane_kind,
            "subphase_name": self.subphase_name,
            "resource_id": self.resource_id,
            "clock_domain_id": self.clock_domain_id,
            "started_monotonic_ns": self.started_monotonic_ns,
            "finished_monotonic_ns": self.finished_monotonic_ns,
            "status": self.status,
            "timestamps_measured_directly": (
                self.timestamps_measured_directly
            ),
        }


@dataclass(frozen=True)
class CompletedFitIntervalObservation:
    """Closed interval ledger for one completed concurrent-fit observation."""

    observation_id: str
    owner_execution_ids: tuple[str, ...]
    clock_domain_id: str
    observation_started_monotonic_ns: int
    observation_finished_monotonic_ns: int
    status: str
    intervals: tuple[FitLaneInterval, ...]
    content_sha256: str
    schema_version: str = COMPLETED_FIT_INTERVAL_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMPLETED_FIT_INTERVAL_OBSERVATION_SCHEMA:
            raise ValueError("unsupported completed fit interval observation schema")
        observation_id = _nonempty_text(
            self.observation_id,
            label="observation_id",
        )
        clock = _nonempty_text(
            self.clock_domain_id,
            label="clock_domain_id",
        )
        owners = tuple(
            _nonempty_text(value, label="owner_execution_id")
            for value in self.owner_execution_ids
        )
        if not owners or len(owners) != len(set(owners)):
            raise ValueError(
                "completed fit observation owner IDs must be nonempty and unique"
            )
        start = _timestamp(
            self.observation_started_monotonic_ns,
            label="observation_started_monotonic_ns",
        )
        finish = _timestamp(
            self.observation_finished_monotonic_ns,
            label="observation_finished_monotonic_ns",
        )
        if finish <= start:
            raise ValueError("completed fit observation must have positive duration")
        if self.status != "completed":
            raise ValueError("fit interval observation must be completed")
        intervals = tuple(self.intervals)
        if not intervals or any(
            not isinstance(value, FitLaneInterval) for value in intervals
        ):
            raise TypeError(
                "completed fit observation requires typed interval records"
            )
        interval_ids = tuple(value.interval_id for value in intervals)
        if len(interval_ids) != len(set(interval_ids)):
            raise ValueError("completed fit interval IDs are duplicated")
        observed_owners = {value.owner_execution_id for value in intervals}
        if observed_owners != set(owners):
            raise ValueError(
                "completed fit intervals do not exactly cover declared owner IDs"
            )
        if {value.lane_kind for value in intervals} != _LANES:
            raise ValueError(
                "completed fit observation requires nonempty CPU and GPU intervals"
            )
        if any(
            value.clock_domain_id != clock
            or value.started_monotonic_ns < start
            or value.finished_monotonic_ns > finish
            for value in intervals
        ):
            raise ValueError(
                "fit intervals changed clock domain or escaped the observation window"
            )
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(self, "owner_execution_ids", owners)
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(
            self,
            "observation_started_monotonic_ns",
            start,
        )
        object.__setattr__(
            self,
            "observation_finished_monotonic_ns",
            finish,
        )
        object.__setattr__(self, "intervals", intervals)
        if _SHA256.fullmatch(str(self.content_sha256)) is None:
            raise ValueError(
                "completed fit interval observation content identity is invalid"
            )
        body = self._body()
        if _sha256_json(body) != self.content_sha256:
            raise ValueError(
                "completed fit interval observation content identity changed"
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "owner_execution_ids": list(
                self.owner_execution_ids
            ),
            "clock_domain_id": self.clock_domain_id,
            "observation_started_monotonic_ns": (
                self.observation_started_monotonic_ns
            ),
            "observation_finished_monotonic_ns": (
                self.observation_finished_monotonic_ns
            ),
            "status": self.status,
            "intervals": [value.as_dict() for value in self.intervals],
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CompletedFitIntervalObservation":
        if not isinstance(value, Mapping):
            raise TypeError("completed fit interval observation must be one mapping")
        required = {
            "schema_version",
            "observation_id",
            "owner_execution_ids",
            "clock_domain_id",
            "observation_started_monotonic_ns",
            "observation_finished_monotonic_ns",
            "status",
            "intervals",
            "content_sha256",
        }
        if set(value) != required:
            raise ValueError(
                "completed fit interval observation fields must be closed; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        intervals = value["intervals"]
        if isinstance(intervals, (str, bytes)) or not isinstance(
            intervals,
            Sequence,
        ):
            raise TypeError("completed fit observation intervals must be a sequence")
        owners = value["owner_execution_ids"]
        if isinstance(owners, (str, bytes)) or not isinstance(owners, Sequence):
            raise TypeError("completed fit observation owners must be a sequence")
        return cls(
            schema_version=value["schema_version"],
            observation_id=value["observation_id"],
            owner_execution_ids=tuple(owners),
            clock_domain_id=value["clock_domain_id"],
            observation_started_monotonic_ns=value[
                "observation_started_monotonic_ns"
            ],
            observation_finished_monotonic_ns=value[
                "observation_finished_monotonic_ns"
            ],
            status=value["status"],
            intervals=tuple(
                FitLaneInterval.from_mapping(row) for row in intervals
            ),
            content_sha256=value["content_sha256"],
        )

    @classmethod
    def seal(
        cls,
        *,
        observation_id: str,
        owner_execution_ids: Sequence[str],
        clock_domain_id: str,
        observation_started_monotonic_ns: int,
        observation_finished_monotonic_ns: int,
        intervals: Sequence[FitLaneInterval],
    ) -> "CompletedFitIntervalObservation":
        if isinstance(owner_execution_ids, (str, bytes)) or not isinstance(
            owner_execution_ids,
            Sequence,
        ):
            raise TypeError("owner execution IDs must be one sequence")
        if isinstance(intervals, (str, bytes)) or not isinstance(
            intervals,
            Sequence,
        ):
            raise TypeError("fit lane intervals must be one sequence")
        closed_intervals = tuple(intervals)
        if any(
            not isinstance(value, FitLaneInterval)
            for value in closed_intervals
        ):
            raise TypeError("fit lane intervals must use the typed contract")
        body = {
            "schema_version": COMPLETED_FIT_INTERVAL_OBSERVATION_SCHEMA,
            "observation_id": observation_id,
            "owner_execution_ids": list(owner_execution_ids),
            "clock_domain_id": clock_domain_id,
            "observation_started_monotonic_ns": (
                observation_started_monotonic_ns
            ),
            "observation_finished_monotonic_ns": (
                observation_finished_monotonic_ns
            ),
            "status": "completed",
            "intervals": [
                value.as_dict() for value in closed_intervals
            ],
        }
        return cls(
            observation_id=observation_id,
            owner_execution_ids=tuple(owner_execution_ids),
            clock_domain_id=clock_domain_id,
            observation_started_monotonic_ns=(
                observation_started_monotonic_ns
            ),
            observation_finished_monotonic_ns=(
                observation_finished_monotonic_ns
            ),
            status="completed",
            intervals=closed_intervals,
            content_sha256=_sha256_json(body),
        )


@dataclass(frozen=True)
class CpuGpuLaneOverlapAnalysis:
    observation_id: str
    source_observation_content_sha256: str
    observation_window_duration_ns: int
    any_lane_active_duration_ns: int
    cpu_active_duration_ns: int
    gpu_active_duration_ns: int
    cpu_gpu_overlap_duration_ns: int
    within_owner_overlap_duration_ns: int
    cross_owner_overlap_duration_ns: int
    within_owner_only_overlap_duration_ns: int
    cross_owner_only_overlap_duration_ns: int
    simultaneous_within_and_cross_overlap_duration_ns: int
    cpu_gpu_overlap_fraction_of_observation_window: float
    cpu_gpu_overlap_fraction_of_any_lane_active: float
    cpu_gpu_overlap_fraction_of_cpu_active: float
    cpu_gpu_overlap_fraction_of_gpu_active: float
    within_owner_fraction_of_cpu_gpu_overlap: float
    cross_owner_fraction_of_cpu_gpu_overlap: float
    per_owner_within_overlap_duration_ns: Mapping[str, int]
    descriptive_overlap_only: bool = True
    causal_speedup_claimed: bool = False
    throughput_speedup_estimated: bool = False
    overlap_categories_are_nonadditive: bool = True
    schema_version: str = CPU_GPU_LANE_OVERLAP_ANALYSIS_SCHEMA
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CPU_GPU_LANE_OVERLAP_ANALYSIS_SCHEMA:
            raise ValueError("unsupported CPU/GPU lane-overlap analysis schema")
        _nonempty_text(self.observation_id, label="observation_id")
        if _SHA256.fullmatch(str(self.source_observation_content_sha256)) is None:
            raise ValueError("lane-overlap source identity is invalid")
        durations = (
            self.observation_window_duration_ns,
            self.any_lane_active_duration_ns,
            self.cpu_active_duration_ns,
            self.gpu_active_duration_ns,
            self.cpu_gpu_overlap_duration_ns,
            self.within_owner_overlap_duration_ns,
            self.cross_owner_overlap_duration_ns,
            self.within_owner_only_overlap_duration_ns,
            self.cross_owner_only_overlap_duration_ns,
            self.simultaneous_within_and_cross_overlap_duration_ns,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in durations
        ):
            raise ValueError("lane-overlap durations must be nonnegative integers")
        (
            window,
            any_lane,
            cpu,
            gpu,
            overlap,
            within,
            cross,
            within_only,
            cross_only,
            within_and_cross,
        ) = durations
        if (
            window < any_lane
            or any_lane < max(cpu, gpu)
            or overlap > min(cpu, gpu)
            or overlap
            != within_only + cross_only + within_and_cross
            or within != within_only + within_and_cross
            or cross != cross_only + within_and_cross
        ):
            raise ValueError("lane-overlap duration relationships are invalid")
        fractions = (
            self.cpu_gpu_overlap_fraction_of_observation_window,
            self.cpu_gpu_overlap_fraction_of_any_lane_active,
            self.cpu_gpu_overlap_fraction_of_cpu_active,
            self.cpu_gpu_overlap_fraction_of_gpu_active,
            self.within_owner_fraction_of_cpu_gpu_overlap,
            self.cross_owner_fraction_of_cpu_gpu_overlap,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in fractions
        ):
            raise ValueError("lane-overlap fractions must be finite unit values")
        expected_fractions = (
            _fraction(overlap, window),
            _fraction(overlap, any_lane),
            _fraction(overlap, cpu),
            _fraction(overlap, gpu),
            _fraction(within, overlap),
            _fraction(cross, overlap),
        )
        if any(
            float(observed) != expected
            for observed, expected in zip(
                fractions,
                expected_fractions,
                strict=True,
            )
        ):
            raise ValueError("lane-overlap fractions differ from exact durations")
        if (
            self.descriptive_overlap_only is not True
            or self.causal_speedup_claimed is not False
            or self.throughput_speedup_estimated is not False
            or self.overlap_categories_are_nonadditive is not True
        ):
            raise ValueError(
                "lane-overlap analysis cannot claim causal or throughput speedup"
            )
        per_owner = dict(self.per_owner_within_overlap_duration_ns)
        if any(
            not isinstance(owner, str)
            or not owner
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 0
            for owner, duration in per_owner.items()
        ):
            raise ValueError("per-owner lane-overlap durations are invalid")
        object.__setattr__(
            self,
            "per_owner_within_overlap_duration_ns",
            {
                owner: per_owner[owner] for owner in sorted(per_owner)
            },
        )
        object.__setattr__(
            self,
            "content_sha256",
            _sha256_json(self._body()),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "source_observation_content_sha256": (
                self.source_observation_content_sha256
            ),
            "observation_window_duration_ns": (
                self.observation_window_duration_ns
            ),
            "any_lane_active_duration_ns": self.any_lane_active_duration_ns,
            "cpu_active_duration_ns": self.cpu_active_duration_ns,
            "gpu_active_duration_ns": self.gpu_active_duration_ns,
            "cpu_gpu_overlap_duration_ns": self.cpu_gpu_overlap_duration_ns,
            "within_owner_overlap_duration_ns": (
                self.within_owner_overlap_duration_ns
            ),
            "cross_owner_overlap_duration_ns": (
                self.cross_owner_overlap_duration_ns
            ),
            "within_owner_only_overlap_duration_ns": (
                self.within_owner_only_overlap_duration_ns
            ),
            "cross_owner_only_overlap_duration_ns": (
                self.cross_owner_only_overlap_duration_ns
            ),
            "simultaneous_within_and_cross_overlap_duration_ns": (
                self.simultaneous_within_and_cross_overlap_duration_ns
            ),
            "cpu_gpu_overlap_fraction_of_observation_window": (
                self.cpu_gpu_overlap_fraction_of_observation_window
            ),
            "cpu_gpu_overlap_fraction_of_any_lane_active": (
                self.cpu_gpu_overlap_fraction_of_any_lane_active
            ),
            "cpu_gpu_overlap_fraction_of_cpu_active": (
                self.cpu_gpu_overlap_fraction_of_cpu_active
            ),
            "cpu_gpu_overlap_fraction_of_gpu_active": (
                self.cpu_gpu_overlap_fraction_of_gpu_active
            ),
            "within_owner_fraction_of_cpu_gpu_overlap": (
                self.within_owner_fraction_of_cpu_gpu_overlap
            ),
            "cross_owner_fraction_of_cpu_gpu_overlap": (
                self.cross_owner_fraction_of_cpu_gpu_overlap
            ),
            "per_owner_within_overlap_duration_ns": dict(
                self.per_owner_within_overlap_duration_ns
            ),
            "descriptive_overlap_only": self.descriptive_overlap_only,
            "causal_speedup_claimed": self.causal_speedup_claimed,
            "throughput_speedup_estimated": (
                self.throughput_speedup_estimated
            ),
            "overlap_categories_are_nonadditive": (
                self.overlap_categories_are_nonadditive
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}


def analyze_completed_fit_lane_overlap(
    observation: CompletedFitIntervalObservation | Mapping[str, Any],
    *,
    expected_observation_id: str,
    expected_owner_execution_ids: Sequence[str],
) -> CpuGpuLaneOverlapAnalysis:
    """Compute exact descriptive overlap from one authenticated observation."""

    closed = (
        observation
        if isinstance(observation, CompletedFitIntervalObservation)
        else CompletedFitIntervalObservation.from_mapping(observation)
    )
    expected_id = _nonempty_text(
        expected_observation_id,
        label="expected_observation_id",
    )
    if isinstance(expected_owner_execution_ids, (str, bytes)):
        raise TypeError("expected owner IDs must be one sequence")
    expected_owners = tuple(
        _nonempty_text(value, label="expected_owner_execution_id")
        for value in expected_owner_execution_ids
    )
    if (
        closed.observation_id != expected_id
        or closed.owner_execution_ids != expected_owners
    ):
        raise ValueError(
            "completed fit interval observation differs from the expected "
            "observation or ordered owner IDs"
        )

    events: dict[int, dict[str, Counter[str]]] = defaultdict(
        lambda: {"cpu": Counter(), "gpu": Counter()}
    )
    for interval in closed.intervals:
        events[interval.started_monotonic_ns][interval.lane_kind][
            interval.owner_execution_id
        ] += 1
        events[interval.finished_monotonic_ns][interval.lane_kind][
            interval.owner_execution_id
        ] -= 1
    events[closed.observation_started_monotonic_ns]
    events[closed.observation_finished_monotonic_ns]
    times = sorted(events)
    active = {"cpu": Counter(), "gpu": Counter()}
    totals = Counter()
    per_owner = Counter({owner: 0 for owner in expected_owners})

    for index, timestamp in enumerate(times):
        for lane in _LANES:
            for owner, delta in events[timestamp][lane].items():
                active[lane][owner] += delta
                if active[lane][owner] < 0:
                    raise RuntimeError(
                        "fit lane interval event ledger underflowed"
                    )
                if active[lane][owner] == 0:
                    del active[lane][owner]
        if index + 1 == len(times):
            continue
        duration = times[index + 1] - timestamp
        if duration <= 0:
            raise RuntimeError("fit lane interval event order is invalid")
        cpu_owners = set(active["cpu"])
        gpu_owners = set(active["gpu"])
        cpu_active = bool(cpu_owners)
        gpu_active = bool(gpu_owners)
        if cpu_active or gpu_active:
            totals["any_lane"] += duration
        if cpu_active:
            totals["cpu"] += duration
        if gpu_active:
            totals["gpu"] += duration
        if not (cpu_active and gpu_active):
            continue
        totals["overlap"] += duration
        same = cpu_owners & gpu_owners
        within = bool(same)
        cross = any(
            cpu_owner != gpu_owner
            for cpu_owner in cpu_owners
            for gpu_owner in gpu_owners
        )
        if within:
            totals["within"] += duration
            for owner in same:
                per_owner[owner] += duration
        if cross:
            totals["cross"] += duration
        if within and cross:
            totals["within_and_cross"] += duration
        elif within:
            totals["within_only"] += duration
        elif cross:
            totals["cross_only"] += duration
        else:
            raise RuntimeError("CPU/GPU overlap lacks an owner relationship")

    if active["cpu"] or active["gpu"]:
        raise RuntimeError("fit lane interval event ledger did not close")
    overlap = int(totals["overlap"])
    window = (
        closed.observation_finished_monotonic_ns
        - closed.observation_started_monotonic_ns
    )
    return CpuGpuLaneOverlapAnalysis(
        observation_id=closed.observation_id,
        source_observation_content_sha256=closed.content_sha256,
        observation_window_duration_ns=window,
        any_lane_active_duration_ns=int(totals["any_lane"]),
        cpu_active_duration_ns=int(totals["cpu"]),
        gpu_active_duration_ns=int(totals["gpu"]),
        cpu_gpu_overlap_duration_ns=overlap,
        within_owner_overlap_duration_ns=int(totals["within"]),
        cross_owner_overlap_duration_ns=int(totals["cross"]),
        within_owner_only_overlap_duration_ns=int(totals["within_only"]),
        cross_owner_only_overlap_duration_ns=int(totals["cross_only"]),
        simultaneous_within_and_cross_overlap_duration_ns=int(
            totals["within_and_cross"]
        ),
        cpu_gpu_overlap_fraction_of_observation_window=_fraction(
            overlap,
            window,
        ),
        cpu_gpu_overlap_fraction_of_any_lane_active=_fraction(
            overlap,
            int(totals["any_lane"]),
        ),
        cpu_gpu_overlap_fraction_of_cpu_active=_fraction(
            overlap,
            int(totals["cpu"]),
        ),
        cpu_gpu_overlap_fraction_of_gpu_active=_fraction(
            overlap,
            int(totals["gpu"]),
        ),
        within_owner_fraction_of_cpu_gpu_overlap=_fraction(
            int(totals["within"]),
            overlap,
        ),
        cross_owner_fraction_of_cpu_gpu_overlap=_fraction(
            int(totals["cross"]),
            overlap,
        ),
        per_owner_within_overlap_duration_ns=dict(per_owner),
    )


def analyze_completed_fit_observations_lane_overlap(
    observations: Sequence[
        CompletedFitIntervalObservation | Mapping[str, Any]
    ],
    *,
    expected_observation_owner_execution_bindings: Mapping[
        str,
        Sequence[str],
    ],
) -> tuple[CpuGpuLaneOverlapAnalysis, ...]:
    """Analyze observations independently without merging clock windows."""

    if isinstance(observations, (str, bytes)) or not isinstance(
        observations,
        Sequence,
    ):
        raise TypeError("completed fit observations must be one sequence")
    if not isinstance(
        expected_observation_owner_execution_bindings,
        Mapping,
    ):
        raise TypeError("expected observation-owner bindings must be a mapping")
    rows = tuple(observations)
    if not rows:
        raise ValueError("at least one completed fit observation is required")
    parsed = tuple(
        row
        if isinstance(row, CompletedFitIntervalObservation)
        else CompletedFitIntervalObservation.from_mapping(row)
        for row in rows
    )
    identifiers = tuple(row.observation_id for row in parsed)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("completed fit observation IDs are duplicated")
    if set(identifiers) != set(
        expected_observation_owner_execution_bindings
    ):
        raise ValueError(
            "completed fit observations differ from expected observation coverage"
        )
    return tuple(
        analyze_completed_fit_lane_overlap(
            row,
            expected_observation_id=row.observation_id,
            expected_owner_execution_ids=(
                expected_observation_owner_execution_bindings[
                    row.observation_id
                ]
            ),
        )
        for row in parsed
    )


__all__ = [
    "COMPLETED_FIT_INTERVAL_OBSERVATION_SCHEMA",
    "CPU_GPU_LANE_OVERLAP_ANALYSIS_SCHEMA",
    "FIT_LANE_INTERVAL_SCHEMA",
    "CompletedFitIntervalObservation",
    "CpuGpuLaneOverlapAnalysis",
    "FitLaneInterval",
    "analyze_completed_fit_lane_overlap",
    "analyze_completed_fit_observations_lane_overlap",
]
