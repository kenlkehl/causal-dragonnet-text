"""Measured resource benchmark for the role-neutral Stage 1 execution seam.

The benchmark is deliberately separate from productive Stage 1.  A caller
supplies one already-prepared, one-physical-owner workload for every configured
representative scope.  Every candidate then executes the same configured number
of independent complete workloads at fresh roots.  Candidate concurrency only
changes how those identical workloads are scheduled across discovered devices.

This module contains no cohort-size, candidate-name, or device-ID defaults.
Those operational choices belong to :class:`RoleNeutralBenchmarkConfig`.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from . import performance_telemetry as telemetry_module
from .compact_preflight_compression_benchmark import (
    CompactPreflightCompressionBenchmarkConfig,
    run_compact_preflight_compression_benchmark,
    validate_compact_preflight_compression_benchmark_result,
)
from .performance_telemetry import (
    BenchmarkRunObservation,
    ImmutableInputObservation,
    PerformanceAcceptancePolicy,
    RepresentativeScope,
    SubphaseTelemetry,
    TelemetryLedger,
    _candidate_benchmark_summaries,
)
from .portable_resource_scheduler import (
    GPUResource,
    ResourceInventory,
    ResourcePlan,
    discover_resources,
)
from .portable_workflow_spec import (
    ResourcePerformanceSafetyPolicy,
    identity_sha256,
)
from .stage1_execution_topology_policy import (
    ONE_CONTEXT_PER_SELECTED_DEVICE,
    ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
    Stage1ExecutionTopologyPolicy,
)
from .production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN,
    RoleNeutralProducerFactories,
    RoleNeutralStage1ExecutionPolicy,
    execute_and_publish_role_neutral_stage1,
    validate_role_neutral_component_execution_intervals,
    validate_role_neutral_stage1_execution,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan
from .production_stage1_cluster_preflight_artifact_v2 import (
    PortableProductionStage1ClusterPreflightArtifact,
)
from .role_neutral_all_ten_binding import EXPECTED_COMPONENT_FAMILIES
from .role_neutral_htr_group_execution import (
    RoleNeutralHTROperationalControls,
)
from .role_neutral_lane_overlap_analysis import (
    CompletedFitIntervalObservation,
    FitLaneInterval,
    analyze_completed_fit_lane_overlap,
)
from .neural_numerical_replay import (
    NEURAL_REPLAY_COMPARISON_POLICY,
    validate_neural_replay_settings,
)

ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA = (
    "portable_role_neutral_performance_benchmark_config_v5"
)
ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA = (
    "portable_role_neutral_performance_benchmark_result_v6"
)
ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA = (
    "portable_role_neutral_benchmark_execution_schedule_v2"
)
ROLE_NEUTRAL_BENCHMARK_MATRIX_COVERAGE_SCHEMA = (
    "portable_role_neutral_benchmark_matrix_coverage_v4"
)
ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA = (
    "portable_role_neutral_benchmark_source_binding_v2"
)
ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA = (
    "portable_role_neutral_benchmark_workload_binding_v1"
)
ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA = (
    "portable_role_neutral_benchmark_request_v2"
)
ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA = (
    "portable_role_neutral_benchmark_observation_checkpoint_v2"
)
ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA = (
    "portable_role_neutral_benchmark_interrupted_observation_v1"
)
ROLE_NEUTRAL_BENCHMARK_PAUSED_RESULT_SCHEMA = (
    "portable_role_neutral_benchmark_paused_result_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite positive number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return normalized


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"benchmark JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _safe_output_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty")
    normalized = value.strip()
    component = PurePosixPath(normalized)
    if (
        "\x00" in normalized
        or component.is_absolute()
        or component.parts != (normalized,)
        or normalized in {".", ".."}
    ):
        raise ValueError(f"{label} must be one traversal-safe path component")
    return normalized


@dataclass(frozen=True)
class RoleNeutralBenchmarkScope:
    """One opaque representative size and a fixed comparable work count."""

    label: str
    fit_row_count: int
    fits_per_observation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            _safe_output_component(self.label, label="benchmark scope label"),
        )
        object.__setattr__(
            self,
            "fit_row_count",
            _positive_integer(
                self.fit_row_count,
                label=f"benchmark scope {self.label!r} fit_row_count",
            ),
        )
        object.__setattr__(
            self,
            "fits_per_observation",
            _positive_integer(
                self.fits_per_observation,
                label=f"benchmark scope {self.label!r} fits_per_observation",
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleNeutralBenchmarkScope":
        required = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "benchmark scope must configure every field exactly; "
                f"required={sorted(required)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RoleNeutralBenchmarkCandidate:
    """One device-count/concurrency candidate with no physical device IDs."""

    name: str
    accelerator_count: int
    concurrency_per_device: int
    host_cpu_budget: int
    executor_mode: str
    neural_query_topology: Stage1ExecutionTopologyPolicy
    htr_operational_controls: RoleNeutralHTROperationalControls

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _safe_output_component(self.name, label="benchmark candidate name"),
        )
        object.__setattr__(
            self,
            "accelerator_count",
            _nonnegative_integer(
                self.accelerator_count,
                label=f"candidate {self.name!r} accelerator_count",
            ),
        )
        object.__setattr__(
            self,
            "concurrency_per_device",
            _positive_integer(
                self.concurrency_per_device,
                label=f"candidate {self.name!r} concurrency_per_device",
            ),
        )
        object.__setattr__(
            self,
            "host_cpu_budget",
            _positive_integer(
                self.host_cpu_budget,
                label=f"candidate {self.name!r} host_cpu_budget",
            ),
        )
        if self.executor_mode not in {
            "fresh_per_fit",
            "persistent_slots",
        }:
            raise ValueError(
                f"candidate {self.name!r} executor_mode must be "
                "fresh_per_fit or persistent_slots"
            )
        if not isinstance(
            self.neural_query_topology,
            Stage1ExecutionTopologyPolicy,
        ):
            raise TypeError(
                f"candidate {self.name!r} requires a typed neural-query topology"
            )
        self.neural_query_topology.effective_parallel_owners_for_shape(
            resource_kind=(
                "cpu" if self.accelerator_count == 0 else "accelerator"
            ),
            device_count=(self.accelerator_count or 1),
            workers_per_device=self.concurrency_per_device,
        )
        if not isinstance(
            self.htr_operational_controls,
            RoleNeutralHTROperationalControls,
        ):
            raise TypeError(
                f"candidate {self.name!r} requires typed HTR operational controls"
            )
        if self.total_concurrency > self.host_cpu_budget:
            raise ValueError(
                f"candidate {self.name!r} concurrency exceeds its host CPU budget"
            )

    @property
    def total_concurrency(self) -> int:
        return self.neural_query_topology.effective_parallel_owners_for_shape(
            resource_kind=(
                "cpu" if self.accelerator_count == 0 else "accelerator"
            ),
            device_count=(self.accelerator_count or 1),
            workers_per_device=self.concurrency_per_device,
        )

    @property
    def resource_slot_count(self) -> int:
        return int(self.accelerator_count or 1) * int(
            self.concurrency_per_device
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralBenchmarkCandidate":
        required = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "benchmark candidate must configure every field exactly; "
                f"required={sorted(required)}"
            )
        payload = dict(value)
        payload["neural_query_topology"] = (
            Stage1ExecutionTopologyPolicy.from_mapping(
                payload["neural_query_topology"]
            )
        )
        payload["htr_operational_controls"] = (
            RoleNeutralHTROperationalControls.from_mapping(
                payload["htr_operational_controls"]
            )
        )
        return cls(**payload)


def _matched_htr_operational_pairs(
    candidates: Sequence[RoleNeutralBenchmarkCandidate],
    *,
    varied_fields: frozenset[str],
    required_difference: str,
) -> tuple[tuple[str, str], ...]:
    """Derive resource-matched pairs differing in one operational factor."""

    values = tuple(candidates)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if (
                left.accelerator_count,
                left.concurrency_per_device,
                left.host_cpu_budget,
                left.executor_mode,
                left.neural_query_topology,
            ) != (
                right.accelerator_count,
                right.concurrency_per_device,
                right.host_cpu_budget,
                right.executor_mode,
                right.neural_query_topology,
            ):
                continue
            left_controls = left.htr_operational_controls.as_dict()
            right_controls = right.htr_operational_controls.as_dict()
            if {
                key: value
                for key, value in left_controls.items()
                if key not in varied_fields
            } != {
                key: value
                for key, value in right_controls.items()
                if key not in varied_fields
            }:
                continue
            if (
                left_controls[required_difference]
                == right_controls[required_difference]
            ):
                continue
            pairs.append(tuple(sorted((left.name, right.name))))
    return tuple(sorted(set(pairs)))


def _matched_neural_query_topology_pairs(
    candidates: Sequence[RoleNeutralBenchmarkCandidate],
) -> tuple[tuple[str, str], ...]:
    """Return resource/control-matched candidates differing only in topology."""

    values = tuple(candidates)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if (
                left.accelerator_count,
                left.concurrency_per_device,
                left.host_cpu_budget,
                left.executor_mode,
                left.htr_operational_controls,
            ) != (
                right.accelerator_count,
                right.concurrency_per_device,
                right.host_cpu_budget,
                right.executor_mode,
                right.htr_operational_controls,
            ):
                continue
            if left.neural_query_topology == right.neural_query_topology:
                continue
            pairs.append(tuple(sorted((left.name, right.name))))
    return tuple(sorted(set(pairs)))


@dataclass(frozen=True)
class RoleNeutralBenchmarkConfig:
    """Complete operational benchmark matrix loaded from deployment JSON."""

    representative_scopes: tuple[RoleNeutralBenchmarkScope, ...]
    candidates: tuple[RoleNeutralBenchmarkCandidate, ...]
    scientific_reference_candidate: str
    multi_device_baselines: tuple[tuple[str, str], ...]
    resource_performance_safety: ResourcePerformanceSafetyPolicy
    preflight_compression_benchmark: (
        CompactPreflightCompressionBenchmarkConfig
    )
    gpu_sample_interval_seconds: float
    warmup_observations_per_candidate_scope: int
    schema_version: str = ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA:
            raise ValueError("unsupported role-neutral benchmark config schema")
        scopes = tuple(self.representative_scopes)
        candidates = tuple(self.candidates)
        if (
            not scopes
            or any(not isinstance(value, RoleNeutralBenchmarkScope) for value in scopes)
            or len({value.label for value in scopes}) != len(scopes)
        ):
            raise ValueError("benchmark requires unique typed representative scopes")
        if (
            not candidates
            or any(
                not isinstance(value, RoleNeutralBenchmarkCandidate)
                for value in candidates
            )
            or len({value.name for value in candidates}) != len(candidates)
        ):
            raise ValueError("benchmark requires unique typed candidates")
        if not isinstance(
            self.resource_performance_safety,
            ResourcePerformanceSafetyPolicy,
        ):
            raise TypeError("benchmark requires typed resource/performance safety")
        if not isinstance(
            self.preflight_compression_benchmark,
            CompactPreflightCompressionBenchmarkConfig,
        ):
            raise TypeError(
                "benchmark requires typed compact-preflight compression settings"
            )
        object.__setattr__(
            self,
            "gpu_sample_interval_seconds",
            _finite_positive(
                self.gpu_sample_interval_seconds,
                label="gpu_sample_interval_seconds",
            ),
        )
        object.__setattr__(
            self,
            "warmup_observations_per_candidate_scope",
            _nonnegative_integer(
                self.warmup_observations_per_candidate_scope,
                label="warmup_observations_per_candidate_scope",
            ),
        )
        object.__setattr__(self, "representative_scopes", scopes)
        object.__setattr__(self, "candidates", candidates)
        training_batches = {
            value.htr_operational_controls.training_batch_size
            for value in candidates
        }
        if len(training_batches) != 1:
            raise ValueError(
                "benchmark candidates cannot vary the scientific HTR "
                "optimizer training batch"
            )
        encoder_batches = {
            value.htr_operational_controls.sentence_encoder_batch_size
            for value in candidates
        }
        data_loader_workers = {
            value.htr_operational_controls.data_loader_workers
            for value in candidates
        }
        reusable_plan_modes = {
            value.htr_operational_controls.reuse_tokenizer_and_chunk_plans
            for value in candidates
        }
        if len(encoder_batches) < 2:
            raise ValueError(
                "benchmark must configure at least two HTR encoder subbatch sizes"
            )
        if not (
            0 in data_loader_workers
            and any(value > 0 for value in data_loader_workers)
        ):
            raise ValueError(
                "benchmark must configure zero and positive HTR data-loader workers"
            )
        if reusable_plan_modes != {False, True}:
            raise ValueError(
                "benchmark must configure disabled and enabled reusable HTR plans"
            )
        reference = str(self.scientific_reference_candidate).strip()
        by_name = {value.name: value for value in candidates}
        if reference not in by_name:
            raise ValueError("scientific reference candidate is not configured")
        object.__setattr__(self, "scientific_reference_candidate", reference)
        baselines = tuple(
            (str(candidate).strip(), str(baseline).strip())
            for candidate, baseline in self.multi_device_baselines
        )
        if (
            len({candidate for candidate, _baseline in baselines}) != len(baselines)
            or any(
                candidate not in by_name
                or baseline not in by_name
                or candidate == baseline
                or by_name[candidate].accelerator_count <= 1
                or by_name[baseline].accelerator_count != 1
                or (
                    by_name[candidate].concurrency_per_device
                    != by_name[baseline].concurrency_per_device
                )
                or (
                    by_name[candidate].host_cpu_budget
                    != by_name[baseline].host_cpu_budget
                )
                or (
                    by_name[candidate].executor_mode
                    != by_name[baseline].executor_mode
                )
                or (
                    by_name[candidate].htr_operational_controls
                    != by_name[baseline].htr_operational_controls
                )
                for candidate, baseline in baselines
            )
        ):
            raise ValueError(
                "benchmark multi-device baseline bindings must preserve "
                "per-device concurrency, CPU budget, executor lifecycle, and "
                "all HTR operational controls"
            )
        configured_multi = {
            value.name for value in candidates if value.accelerator_count > 1
        }
        if {candidate for candidate, _baseline in baselines} != configured_multi:
            raise ValueError(
                "every multi-device candidate requires one single-device baseline"
            )
        object.__setattr__(self, "multi_device_baselines", baselines)
        maximum_concurrency = max(value.total_concurrency for value in candidates)
        if any(
            scope.fits_per_observation < maximum_concurrency
            for scope in scopes
        ):
            raise ValueError(
                "every representative observation must contain enough independent "
                "fits to exercise the largest configured candidate concurrency"
            )
        configured_executor_modes = {
            value.executor_mode for value in candidates
        }
        if configured_executor_modes != {
            "fresh_per_fit",
            "persistent_slots",
        }:
            raise ValueError(
                "benchmark must configure both fresh_per_fit and "
                "persistent_slots executor candidates"
            )
        fresh_shapes = {
            (
                value.accelerator_count,
                value.concurrency_per_device,
                value.host_cpu_budget,
                value.neural_query_topology,
                value.htr_operational_controls,
            )
            for value in candidates
            if value.executor_mode == "fresh_per_fit"
        }
        persistent_shapes = {
            (
                value.accelerator_count,
                value.concurrency_per_device,
                value.host_cpu_budget,
                value.neural_query_topology,
                value.htr_operational_controls,
            )
            for value in candidates
            if value.executor_mode == "persistent_slots"
        }
        if not fresh_shapes.intersection(persistent_shapes):
            raise ValueError(
                "fresh and persistent benchmark modes require at least one "
                "matched resource/concurrency/CPU/HTR-controls candidate pair"
            )

        if not _matched_htr_operational_pairs(
            candidates,
            varied_fields=frozenset({"sentence_encoder_batch_size"}),
            required_difference="sentence_encoder_batch_size",
        ):
            raise ValueError(
                "HTR encoder-subbatch benchmarking requires a matched "
                "resource/concurrency/executor/control pair"
            )
        topology_modes = {
            candidate.neural_query_topology.mode
            for candidate in candidates
        }
        multi_accelerator_available = any(
            candidate.accelerator_count > 1
            for candidate in candidates
        )
        if multi_accelerator_available:
            if topology_modes != {
                ONE_CONTEXT_PER_SELECTED_DEVICE,
                ONE_CONTEXT_SPANNING_ALL_SELECTED_DEVICES,
            }:
                raise ValueError(
                    "a multi-accelerator benchmark must configure per-device "
                    "and spanning learned-query topology modes"
                )
            if not _matched_neural_query_topology_pairs(candidates):
                raise ValueError(
                    "learned-query topology benchmarking requires a matched "
                    "resource/concurrency/executor/HTR-control pair"
                )
        elif topology_modes != {ONE_CONTEXT_PER_SELECTED_DEVICE}:
            raise ValueError(
                "CPU or single-accelerator benchmark candidates cannot claim "
                "a spanning learned-query topology"
            )
        if not _matched_htr_operational_pairs(
            candidates,
            varied_fields=frozenset({"data_loader_workers"}),
            required_difference="data_loader_workers",
        ):
            raise ValueError(
                "HTR data-loader benchmarking requires a matched "
                "resource/concurrency/executor/control pair"
            )
        if not _matched_htr_operational_pairs(
            candidates,
            varied_fields=frozenset(
                {
                    "reuse_tokenizer_and_chunk_plans",
                    "chunk_plan_cache_max_entries",
                    "tokenized_chunk_cache_max_entries",
                }
            ),
            required_difference="reuse_tokenizer_and_chunk_plans",
        ):
            raise ValueError(
                "HTR reusable-plan benchmarking requires a matched "
                "resource/concurrency/executor/control pair"
            )

    @property
    def acceptance_policy(self) -> PerformanceAcceptancePolicy:
        return PerformanceAcceptancePolicy(
            representative_scopes=tuple(
                RepresentativeScope(
                    label=value.label,
                    fit_row_count=value.fit_row_count,
                )
                for value in self.representative_scopes
            ),
            resource_performance_safety=self.resource_performance_safety,
            scientific_reference_candidate=self.scientific_reference_candidate,
            multi_device_baselines=self.multi_device_baselines,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "representative_scopes": [
                asdict(value) for value in self.representative_scopes
            ],
            "candidates": [asdict(value) for value in self.candidates],
            "scientific_reference_candidate": self.scientific_reference_candidate,
            "multi_device_baselines": [
                {"candidate": candidate, "baseline": baseline}
                for candidate, baseline in self.multi_device_baselines
            ],
            "resource_performance_safety": (
                self.resource_performance_safety.as_dict()
            ),
            "preflight_compression_benchmark": (
                self.preflight_compression_benchmark.as_dict()
            ),
            "gpu_sample_interval_seconds": self.gpu_sample_interval_seconds,
            "warmup_observations_per_candidate_scope": (
                self.warmup_observations_per_candidate_scope
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoleNeutralBenchmarkConfig":
        required = {
            "schema_version",
            "representative_scopes",
            "candidates",
            "scientific_reference_candidate",
            "multi_device_baselines",
            "resource_performance_safety",
            "preflight_compression_benchmark",
            "gpu_sample_interval_seconds",
            "warmup_observations_per_candidate_scope",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "benchmark config must configure every field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        raw_scopes = value["representative_scopes"]
        raw_candidates = value["candidates"]
        raw_baselines = value["multi_device_baselines"]
        if (
            not isinstance(raw_scopes, list)
            or not isinstance(raw_candidates, list)
            or not isinstance(raw_baselines, list)
        ):
            raise TypeError("benchmark scopes, candidates, and baselines must be lists")
        baselines: list[tuple[str, str]] = []
        for row in raw_baselines:
            if not isinstance(row, Mapping) or set(row) != {"candidate", "baseline"}:
                raise ValueError("benchmark baseline rows are invalid")
            baselines.append((str(row["candidate"]), str(row["baseline"])))
        safety = value["resource_performance_safety"]
        compression = value["preflight_compression_benchmark"]
        if not isinstance(safety, Mapping) or not isinstance(
            compression,
            Mapping,
        ):
            raise TypeError(
                "benchmark safety and compression policies must be mappings"
            )
        return cls(
            schema_version=str(value["schema_version"]),
            representative_scopes=tuple(
                RoleNeutralBenchmarkScope.from_mapping(row)
                for row in raw_scopes
            ),
            candidates=tuple(
                RoleNeutralBenchmarkCandidate.from_mapping(row)
                for row in raw_candidates
            ),
            scientific_reference_candidate=str(
                value["scientific_reference_candidate"]
            ),
            multi_device_baselines=tuple(baselines),
            resource_performance_safety=(
                ResourcePerformanceSafetyPolicy.from_mapping(safety)
            ),
            preflight_compression_benchmark=(
                CompactPreflightCompressionBenchmarkConfig.from_mapping(
                    compression
                )
            ),
            gpu_sample_interval_seconds=value["gpu_sample_interval_seconds"],
            warmup_observations_per_candidate_scope=(
                value["warmup_observations_per_candidate_scope"]
            ),
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "RoleNeutralBenchmarkConfig":
        source = Path(path)
        try:
            value = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"benchmark JSON contains {constant}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("benchmark config is not closed UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("benchmark config must contain one JSON object")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class RoleNeutralBenchmarkSourceBinding:
    """Immutable staged-workflow authority shared by every measured workload."""

    workflow_request_sha256: str
    workflow_scientific_sha256: str
    workload_deployment_sha256: str
    stage1_preflight_phase_content_sha256: str
    prepared_stage1_context_content_root_sha256: str
    schema_version: str = ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA:
            raise ValueError("unsupported benchmark source-binding schema")
        for field_name in (
            "workflow_request_sha256",
            "workflow_scientific_sha256",
            "workload_deployment_sha256",
            "stage1_preflight_phase_content_sha256",
            "prepared_stage1_context_content_root_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, field_name))) is None:
                raise ValueError(
                    f"benchmark source binding {field_name} must be one "
                    "lowercase SHA-256"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RoleNeutralBenchmarkWorkload:
    """Prepared real role-neutral work for one representative scope."""

    scope_label: str
    plan: Stage1ScopePlan
    scientific_htr_training_batch_size: int
    producer_factories_builder: Callable[[], RoleNeutralProducerFactories]
    physical_owner_executor_builder: Callable[[str, int], Any]
    preflight_compression_source_builder: Callable[
        [],
        PortableProductionStage1ClusterPreflightArtifact,
    ]
    immutable_inputs: tuple[ImmutableInputObservation, ...]
    source_binding: RoleNeutralBenchmarkSourceBinding

    def __post_init__(self) -> None:
        if not isinstance(self.scope_label, str) or not self.scope_label.strip():
            raise ValueError("benchmark workload scope_label must be nonempty")
        object.__setattr__(self, "scope_label", self.scope_label.strip())
        if not isinstance(self.plan, Stage1ScopePlan):
            raise TypeError("benchmark workload requires a Stage1ScopePlan")
        if len(self.plan.physical_scopes) != 1:
            raise ValueError(
                "representative benchmark workload must contain exactly one "
                "physical owner"
            )
        if (
            isinstance(self.scientific_htr_training_batch_size, bool)
            or not isinstance(self.scientific_htr_training_batch_size, int)
            or self.scientific_htr_training_batch_size < 1
        ):
            raise ValueError(
                "benchmark workload requires the authenticated positive "
                "scientific HTR optimizer batch"
            )
        if not callable(self.producer_factories_builder):
            raise TypeError("benchmark workload requires a producer-factories builder")
        if not callable(self.physical_owner_executor_builder):
            raise TypeError("benchmark workload requires a physical-owner executor builder")
        if not callable(self.preflight_compression_source_builder):
            raise TypeError(
                "benchmark workload requires a compact-preflight source builder"
            )
        if not isinstance(self.source_binding, RoleNeutralBenchmarkSourceBinding):
            raise TypeError("benchmark workload requires a typed staged source binding")
        inputs = tuple(self.immutable_inputs)
        if not inputs or any(
            not isinstance(value, ImmutableInputObservation) for value in inputs
        ):
            raise TypeError("benchmark workload requires typed immutable inputs")
        object.__setattr__(self, "immutable_inputs", inputs)

    @property
    def fit_row_count(self) -> int:
        return self.plan.physical_scopes[0].fit_row_count

    @property
    def unique_immutable_input_bytes(self) -> int:
        sizes: dict[str, int] = {}
        for value in self.immutable_inputs:
            previous = sizes.get(value.content_sha256)
            if previous is not None and previous != value.size_bytes:
                raise ValueError("immutable input identity has conflicting sizes")
            sizes[value.content_sha256] = value.size_bytes
        return sum(sizes.values())


_REQUIRED_MATRIX_AXIS_IDS = (
    "executor_lifecycle_fresh_vs_persistent",
    "htr_scope_concurrency_per_compatible_accelerator",
    "htr_batches_and_data_loader_workers",
    "reusable_tokenizer_and_chunk_plans",
    "neural_query_context_device_topology",
    "compression_and_cpu_gpu_lane_overlap",
)
_MATRIX_CODE_FILES = {
    "role_neutral_performance_benchmark": (
        "role_neutral_performance_benchmark.py"
    ),
    "production_role_neutral_producer_factories": (
        "production_role_neutral_producer_factories.py"
    ),
    "role_neutral_htr_group_execution": (
        "role_neutral_htr_group_execution.py"
    ),
    "scientific_profile_identity": "scientific_profile_identity.py",
    "role_neutral_neural_query_group_execution": (
        "role_neutral_neural_query_group_execution.py"
    ),
    "role_neutral_embedding_group_execution": (
        "role_neutral_embedding_group_execution.py"
    ),
    "role_neutral_tfidf_group_execution": (
        "role_neutral_tfidf_group_execution.py"
    ),
    "compact_preflight_compression_benchmark": (
        "compact_preflight_compression_benchmark.py"
    ),
    "role_neutral_lane_overlap_analysis": (
        "role_neutral_lane_overlap_analysis.py"
    ),
    "production_stage1_cluster_preflight_artifact_v2": (
        "production_stage1_cluster_preflight_artifact_v2.py"
    ),
}
_RESUME_CODE_FILES = {
    "performance_telemetry": "performance_telemetry.py",
    "portable_resource_scheduler": "portable_resource_scheduler.py",
    "production_role_neutral_process_executor": (
        "production_role_neutral_process_executor.py"
    ),
    "production_role_neutral_persistent_executor": (
        "production_role_neutral_persistent_executor.py"
    ),
    "production_stage1_role_neutral_execution": (
        "production_stage1_role_neutral_execution.py"
    ),
    "role_neutral_all_ten_binding": "role_neutral_all_ten_binding.py",
    "role_neutral_benchmark_workload_provider": (
        "role_neutral_benchmark_workload_provider.py"
    ),
}


def _benchmark_matrix_code_evidence() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    result: dict[str, str] = {}
    for module_name, filename in sorted(_MATRIX_CODE_FILES.items()):
        path = root / filename
        before = os.stat(path)
        payload = path.read_bytes()
        after = os.stat(path)
        if (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
            int(before.st_mtime_ns),
        ) != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        ):
            raise RuntimeError(
                "benchmark matrix evidence code changed while hashing"
            )
        result[module_name] = hashlib.sha256(payload).hexdigest()
    return result


def _benchmark_resume_code_evidence() -> dict[str, str]:
    result = _benchmark_matrix_code_evidence()
    root = Path(__file__).resolve().parent
    for module_name, filename in sorted(_RESUME_CODE_FILES.items()):
        path = root / filename
        before = os.stat(path)
        payload = path.read_bytes()
        after = os.stat(path)
        if _stat_identity(before) != _stat_identity(after):
            raise RuntimeError(
                "benchmark resume code changed while hashing"
            )
        result[module_name] = hashlib.sha256(payload).hexdigest()
    return result


def _candidate_scientific_equality(
    *,
    candidate_names: Sequence[str],
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_candidate: str,
) -> bool:
    by_name = {
        str(value.get("candidate_name")): value
        for value in candidate_rows
        if isinstance(value, Mapping)
    }
    if set(candidate_names) - set(by_name):
        return False
    reference = by_name.get(reference_candidate)
    reference_scopes = (
        reference.get("scope_results")
        if isinstance(reference, Mapping)
        else None
    )
    if not isinstance(reference_scopes, list) or not reference_scopes:
        return False
    expected = {
        str(value.get("scope_label")): value.get(
            "scientific_artifact_sha256"
        )
        for value in reference_scopes
        if isinstance(value, Mapping)
        and value.get("deterministic_artifact_identity") is True
        and _SHA256.fullmatch(
            str(value.get("scientific_artifact_sha256", ""))
        )
        is not None
    }
    if len(expected) != len(reference_scopes):
        return False
    for name in candidate_names:
        row = by_name[name]
        scopes = row.get("scope_results")
        if (
            not isinstance(scopes, list)
            or len(scopes) != len(expected)
            or row.get("warmup_observation_telemetry_accepted") is not True
            or row.get("warmup_scientific_identity_matches_measured")
            is not True
            or any(
                not isinstance(value, Mapping)
                or value.get("deterministic_artifact_identity") is not True
                or expected.get(str(value.get("scope_label")))
                != value.get("scientific_artifact_sha256")
                for value in scopes
            )
        ):
            return False
    return True


def build_role_neutral_benchmark_matrix_coverage(
    *,
    config: RoleNeutralBenchmarkConfig,
    candidate_rows: Sequence[Mapping[str, Any]],
    compression_benchmark_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Account for every required v1 performance-matrix axis."""

    if not isinstance(config, RoleNeutralBenchmarkConfig):
        raise TypeError("benchmark matrix coverage requires a typed config")
    rows = tuple(candidate_rows)
    if not rows or any(not isinstance(value, Mapping) for value in rows):
        raise TypeError("benchmark matrix coverage requires candidate rows")
    compression = (
        validate_compact_preflight_compression_benchmark_result(
            compression_benchmark_result,
            reopen_artifacts=False,
        )
    )
    if (
        compression["config"]
        != config.preflight_compression_benchmark.as_dict()
    ):
        raise ValueError(
            "compression benchmark differs from the configured matrix"
        )
    code = _benchmark_matrix_code_evidence()

    def evidence(*names: str) -> list[dict[str, str]]:
        return [
            {
                "module": name,
                "source_sha256": code[name],
            }
            for name in names
        ]

    candidates = tuple(config.candidates)
    lifecycle_names = [value.name for value in candidates]
    lifecycle_equal = _candidate_scientific_equality(
        candidate_names=lifecycle_names,
        candidate_rows=rows,
        reference_candidate=config.scientific_reference_candidate,
    )
    accelerator_concurrency = tuple(
        value for value in candidates if value.accelerator_count == 1
    )
    accelerator_concurrency_values = {
        value.concurrency_per_device for value in accelerator_concurrency
    }
    concurrency_configured = (
        1 in accelerator_concurrency_values
        and any(value >= 2 for value in accelerator_concurrency_values)
    )
    concurrency_equal = (
        concurrency_configured
        and _candidate_scientific_equality(
            candidate_names=[
                value.name for value in accelerator_concurrency
            ],
            candidate_rows=rows,
            reference_candidate=config.scientific_reference_candidate,
        )
    )
    rows_by_name = {
        str(value.get("candidate_name")): value for value in rows
    }
    htr_operational_attested = set(rows_by_name) == set(lifecycle_names) and all(
        rows_by_name[name].get("htr_operational_attestations_accepted")
        is True
        and rows_by_name[name].get("htr_operational_controls")
        == next(
            value.htr_operational_controls.as_dict()
            for value in candidates
            if value.name == name
        )
        for name in lifecycle_names
    )
    encoder_subbatch_pairs = _matched_htr_operational_pairs(
        candidates,
        varied_fields=frozenset({"sentence_encoder_batch_size"}),
        required_difference="sentence_encoder_batch_size",
    )
    data_loader_worker_pairs = _matched_htr_operational_pairs(
        candidates,
        varied_fields=frozenset({"data_loader_workers"}),
        required_difference="data_loader_workers",
    )
    reusable_plan_pairs = _matched_htr_operational_pairs(
        candidates,
        varied_fields=frozenset(
            {
                "reuse_tokenizer_and_chunk_plans",
                "chunk_plan_cache_max_entries",
                "tokenized_chunk_cache_max_entries",
            }
        ),
        required_difference="reuse_tokenizer_and_chunk_plans",
    )
    htr_operational_equal = (
        lifecycle_equal
        and htr_operational_attested
        and bool(encoder_subbatch_pairs)
        and bool(data_loader_worker_pairs)
        and bool(reusable_plan_pairs)
    )
    neural_query_topology_pairs = (
        _matched_neural_query_topology_pairs(candidates)
    )
    neural_query_topology_candidate_names = sorted(
        {
            name
            for pair in neural_query_topology_pairs
            for name in pair
        }
    )
    neural_query_topology_scientific_equal = bool(
        neural_query_topology_pairs
    ) and _candidate_scientific_equality(
        candidate_names=neural_query_topology_candidate_names,
        candidate_rows=rows,
        reference_candidate=config.scientific_reference_candidate,
    )
    neural_query_topology_runtime_attested = (
        neural_query_topology_scientific_equal
        and all(
            rows_by_name[name].get(
                "neural_query_topology_runtime_attestations_accepted"
            )
            is True
            for name in neural_query_topology_candidate_names
        )
    )
    accelerator_candidate_names = [
        value.name
        for value in candidates
        if value.accelerator_count > 0
    ]
    expected_lane_observations_per_candidate = (
        len(config.representative_scopes)
        * config.resource_performance_safety
        .minimum_benchmark_repetitions_per_scope
    )
    lane_overlap_applicable = bool(accelerator_candidate_names)
    lane_overlap_descriptively_measured = (
        lane_overlap_applicable
        and lifecycle_equal
        and all(
            rows_by_name[name].get(
                "cpu_gpu_lane_interval_telemetry_accepted"
            )
            is True
            and rows_by_name[name].get(
                "cpu_gpu_lane_overlap_observation_count"
            )
            == expected_lane_observations_per_candidate
            and rows_by_name[name].get(
                "cpu_gpu_lane_overlap_descriptive_only"
            )
            is True
            and rows_by_name[name].get(
                "cpu_gpu_lane_overlap_speedup_claimed"
            )
            is False
            for name in accelerator_candidate_names
        )
    )
    axes = [
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[0],
            "disposition": (
                "measured" if lifecycle_equal else "equality_rejected"
            ),
            "performance_claimed": lifecycle_equal,
            "configured_candidate_values": [
                {
                    "candidate_name": value.name,
                    "executor_mode": value.executor_mode,
                }
                for value in candidates
            ],
            "equality_gate": (
                "complete_artifacts_equal_to_scientific_reference"
                if lifecycle_equal
                else "complete_artifacts_failed_scientific_equality"
            ),
            "reason_code": "executor_modes_are_deployment_only",
            "component_dispositions": [
                {
                    "component": "fresh_per_fit_vs_persistent_slots",
                    "disposition": (
                        "measured"
                        if lifecycle_equal
                        else "equality_rejected"
                    ),
                }
            ],
            "code_evidence": evidence(
                "role_neutral_performance_benchmark",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[1],
            "disposition": (
                "measured"
                if concurrency_equal
                else (
                    "equality_rejected"
                    if concurrency_configured
                    else "unsupported_by_v1_executor"
                )
            ),
            "performance_claimed": concurrency_equal,
            "configured_candidate_values": [
                {
                    "candidate_name": value.name,
                    "accelerator_count": value.accelerator_count,
                    "concurrency_per_device": (
                        value.concurrency_per_device
                    ),
                }
                for value in accelerator_concurrency
            ],
            "equality_gate": (
                "complete_artifacts_equal_to_scientific_reference"
                if concurrency_equal
                else (
                    "complete_artifacts_failed_scientific_equality"
                    if concurrency_configured
                    else "not_applicable_without_compatible_accelerator_variants"
                )
            ),
            "reason_code": (
                "configured_one_and_multiple_scopes_per_accelerator"
                if concurrency_configured
                else "no_compatible_accelerator_concurrency_matrix"
            ),
            "component_dispositions": [
                {
                    "component": "one_vs_multiple_htr_scopes_per_accelerator",
                    "disposition": (
                        "measured"
                        if concurrency_equal
                        else (
                            "equality_rejected"
                            if concurrency_configured
                            else "unsupported_by_v1_executor"
                        )
                    ),
                }
            ],
            "code_evidence": evidence(
                "role_neutral_performance_benchmark",
                "role_neutral_htr_group_execution",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[2],
            "disposition": (
                "partially_measured"
                if htr_operational_equal
                else "equality_rejected"
            ),
            "performance_claimed": htr_operational_equal,
            "configured_candidate_values": [
                {
                    "candidate_name": value.name,
                    "training_batch_size": (
                        value.htr_operational_controls.training_batch_size
                    ),
                    "sentence_encoder_batch_size": (
                        value.htr_operational_controls
                        .sentence_encoder_batch_size
                    ),
                    "data_loader_workers": (
                        value.htr_operational_controls.data_loader_workers
                    ),
                }
                for value in candidates
            ],
            "equality_gate": (
                "complete_artifacts_equal_and_operationally_attested"
                if htr_operational_equal
                else "complete_artifacts_or_operational_attestations_differ"
            ),
            "reason_code": (
                "optimizer_batch_remains_scientific_encoder_subbatch_and_"
                "data_loader_plan_workers_are_deployment_only"
            ),
            "component_dispositions": [
                {
                    "component": "htr_training_batch_size",
                    "disposition": (
                        "scientific_configuration_not_operationally_tunable"
                    ),
                    "performance_claimed": False,
                    "reason_code": (
                        "optimizer_update_batch_changes_gradient_aggregation"
                    ),
                },
                {
                    "component": "htr_sentence_encoder_batch_size",
                    "disposition": (
                        "measured"
                        if htr_operational_equal
                        else "equality_rejected"
                    ),
                    "performance_claimed": htr_operational_equal,
                    "matched_one_factor_pairs": [
                        {
                            "candidate_a": left,
                            "candidate_b": right,
                        }
                        for left, right in encoder_subbatch_pairs
                    ],
                },
                {
                    "component": "htr_data_loader_workers",
                    "disposition": (
                        "measured"
                        if htr_operational_equal
                        else "equality_rejected"
                    ),
                    "performance_claimed": htr_operational_equal,
                    "matched_one_factor_pairs": [
                        {
                            "candidate_a": left,
                            "candidate_b": right,
                        }
                        for left, right in data_loader_worker_pairs
                    ],
                },
            ],
            "code_evidence": evidence(
                "production_role_neutral_producer_factories",
                "role_neutral_htr_group_execution",
                "scientific_profile_identity",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[3],
            "disposition": (
                "measured"
                if htr_operational_equal
                else "equality_rejected"
            ),
            "performance_claimed": htr_operational_equal,
            "configured_candidate_values": [
                {
                    "candidate_name": value.name,
                    "reuse_tokenizer_and_chunk_plans": (
                        value.htr_operational_controls
                        .reuse_tokenizer_and_chunk_plans
                    ),
                    "chunk_plan_cache_max_entries": (
                        value.htr_operational_controls
                        .chunk_plan_cache_max_entries
                    ),
                    "tokenized_chunk_cache_max_entries": (
                        value.htr_operational_controls
                        .tokenized_chunk_cache_max_entries
                    ),
                }
                for value in candidates
            ],
            "equality_gate": (
                "complete_artifacts_equal_and_row_keyed_plans_authenticated"
                if htr_operational_equal
                else "complete_artifacts_or_reusable_plan_attestations_differ"
            ),
            "reason_code": (
                "row_and_configuration_bound_in_process_plan_reused_across_"
                "htr_folds_without_persisting_raw_text"
            ),
            "matched_one_factor_pairs": [
                {
                    "candidate_a": left,
                    "candidate_b": right,
                }
                for left, right in reusable_plan_pairs
            ],
            "component_dispositions": [
                {
                    "component": "cached_tokenizer",
                    "disposition": (
                        "measured"
                        if htr_operational_equal
                        else "equality_rejected"
                    ),
                },
                {
                    "component": "cached_chunk_plan",
                    "disposition": (
                        "measured"
                        if htr_operational_equal
                        else "equality_rejected"
                    ),
                },
            ],
            "code_evidence": evidence(
                "role_neutral_htr_group_execution",
                "production_role_neutral_producer_factories",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[4],
            "disposition": (
                "measured"
                if neural_query_topology_runtime_attested
                else (
                    "runtime_attestation_unavailable"
                    if neural_query_topology_scientific_equal
                    else (
                        "equality_rejected"
                        if neural_query_topology_pairs
                        else "unsupported_by_available_resources"
                    )
                )
            ),
            "performance_claimed": (
                neural_query_topology_runtime_attested
            ),
            "configured_candidate_values": [
                {
                    "candidate_name": value.name,
                    "accelerator_count": value.accelerator_count,
                    "concurrency_per_device": (
                        value.concurrency_per_device
                    ),
                    "effective_parallel_owners": (
                        value.total_concurrency
                    ),
                    "neural_query_topology": (
                        value.neural_query_topology.as_dict()
                    ),
                }
                for value in candidates
                if value.name
                in neural_query_topology_candidate_names
            ],
            "equality_gate": (
                "complete_artifacts_equal_and_runtime_topology_attested"
                if neural_query_topology_runtime_attested
                else (
                    "complete_artifacts_equal_but_runtime_topology_unattested"
                    if neural_query_topology_scientific_equal
                    else (
                        "complete_artifacts_failed_scientific_equality"
                        if neural_query_topology_pairs
                        else "not_applicable_without_compatible_multi_device_resources"
                    )
                )
            ),
            "reason_code": (
                "atomic_device_tuple_reservation_and_homogeneous_runtime_"
                "topology_attestation"
                if neural_query_topology_pairs
                else "no_compatible_multi_device_topology_matrix"
            ),
            "matched_one_factor_pairs": [
                {
                    "candidate_a": left,
                    "candidate_b": right,
                }
                for left, right in neural_query_topology_pairs
            ],
            "component_dispositions": [
                {
                    "component": (
                        "one_neural_query_context_per_accelerator_vs_"
                        "one_context_spanning_accelerators"
                    ),
                    "disposition": (
                        "measured"
                        if neural_query_topology_runtime_attested
                        else (
                            "runtime_attestation_unavailable"
                            if neural_query_topology_scientific_equal
                            else (
                                "equality_rejected"
                                if neural_query_topology_pairs
                                else "unsupported_by_available_resources"
                            )
                        )
                    ),
                }
            ],
            "code_evidence": evidence(
                "role_neutral_neural_query_group_execution",
                "production_role_neutral_producer_factories",
                "role_neutral_performance_benchmark",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[5],
            "disposition": (
                "partially_measured"
                if (
                    compression["accepted"] is True
                    and lane_overlap_descriptively_measured
                )
                else (
                    "runtime_attestation_unavailable"
                    if (
                        compression["accepted"] is True
                        and not lane_overlap_applicable
                    )
                    else "equality_rejected"
                )
            ),
            "performance_claimed": False,
            "configured_candidate_values": [
                {
                    "parquet_compression": row[
                        "parquet_compression"
                    ],
                    "median_wall_seconds": row[
                        "median_wall_seconds"
                    ],
                    "median_cpu_seconds": row[
                        "median_cpu_seconds"
                    ],
                    "output_tree_file_bytes": row[
                        "output_tree_file_bytes"
                    ],
                }
                for row in compression["codec_results"]
            ]
            + [
                {
                    "candidate_name": name,
                    "component_execution_interval_observation_count": (
                        rows_by_name[name].get(
                            "cpu_gpu_lane_overlap_observation_count"
                        )
                    ),
                    "descriptive_overlap_only": (
                        rows_by_name[name].get(
                            "cpu_gpu_lane_overlap_descriptive_only"
                        )
                    ),
                    "speedup_claimed": False,
                }
                for name in accelerator_candidate_names
            ],
            "equality_gate": (
                "compact_preflight_codecs_equal_and_complete_fit_phase_"
                "intervals_authenticated"
                if (
                    compression["accepted"] is True
                    and lane_overlap_descriptively_measured
                )
                else (
                    "compact_preflight_codecs_equal_but_no_accelerator_"
                    "lane_is_applicable"
                    if (
                        compression["accepted"] is True
                        and not lane_overlap_applicable
                    )
                    else "compact_preflight_or_lane_scientific_equality_failed"
                )
            ),
            "reason_code": (
                "compact_preflight_codec_choice_measured_and_cpu_gpu_"
                "architecture_phase_overlap_measured_descriptively"
                if lane_overlap_descriptively_measured
                else (
                    "cpu_only_candidate_matrix_has_no_gpu_lane"
                    if not lane_overlap_applicable
                    else "component_interval_telemetry_incomplete_or_changed"
                )
            ),
            "component_dispositions": [
                {
                    "component": "artifact_compression_choice",
                    "disposition": (
                        "measured"
                        if compression["accepted"] is True
                        else "equality_rejected"
                    ),
                    "performance_claimed": (
                        compression["accepted"] is True
                    ),
                    "selected_parquet_compression": compression[
                        "selected_parquet_compression"
                    ],
                    "scientific_equality_gate": (
                        "exact_path_neutral_scientific_content"
                    ),
                },
                {
                    "component": "cpu_gpu_lane_overlap",
                    "disposition": (
                        "descriptively_measured"
                        if lane_overlap_descriptively_measured
                        else (
                            "unsupported_by_available_resources"
                            if not lane_overlap_applicable
                            else "equality_rejected"
                        )
                    ),
                    "performance_claimed": False,
                    "interval_semantics": (
                        "direct_monotonic_architecture_phase_envelopes_"
                        "not_kernel_occupancy_v1"
                    ),
                    "causal_speedup_claimed": False,
                    "throughput_speedup_estimated": False,
                },
            ],
            "code_evidence": evidence(
                "role_neutral_performance_benchmark",
                "role_neutral_lane_overlap_analysis",
                "compact_preflight_compression_benchmark",
                "production_stage1_cluster_preflight_artifact_v2",
            ),
        },
    ]
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_MATRIX_COVERAGE_SCHEMA,
        "required_axis_ids": list(_REQUIRED_MATRIX_AXIS_IDS),
        "axes": axes,
        "all_required_axes_accounted": (
            [value["axis_id"] for value in axes]
            == list(_REQUIRED_MATRIX_AXIS_IDS)
        ),
        "unsupported_axes_make_no_performance_claim": all(
            value["performance_claimed"] is False
            for value in axes
            if value["disposition"]
            in {
                "scientific_configuration_not_operationally_tunable",
                "unsupported_by_v1_executor",
                "unsupported_by_available_resources",
                "runtime_attestation_unavailable",
            }
        ),
    }
    return {**body, "content_sha256": identity_sha256(body)}


@dataclass(frozen=True)
class _InstanceResult:
    instance_index: int
    root: Path
    device: str
    manifest: Mapping[str, Any] | None
    model_records: tuple[SubphaseTelemetry, ...]
    complete_record: SubphaseTelemetry
    model_wall_seconds: float
    model_cpu_seconds: float
    coordination_wall_seconds: float
    coordination_cpu_seconds: float
    ordinary_process_read_bytes: int
    ordinary_logical_read_bytes: int
    child_process_counters_expected: bool
    child_process_read_bytes: int | None
    child_process_written_bytes: int | None
    child_process_wall_seconds: float | None
    child_process_cpu_seconds: float | None
    child_peak_gpu_memory_bytes_by_device: Mapping[str, int] | None
    child_cpu_budget_attestation: Mapping[str, Any] | None
    htr_operational_attestation: Mapping[str, Any] | None
    neural_query_topology_attestation: Mapping[str, Any] | None
    component_execution_intervals: tuple[Mapping[str, Any], ...]
    peak_allocation_fraction: float | None
    minimum_headroom_bytes: int | None
    gpu_telemetry_complete: bool
    oom_observed: bool

    @property
    def scientific_artifact_sha256(self) -> str | None:
        if self.manifest is None:
            return None
        scientific = self.manifest.get("scientific_identity")
        if not isinstance(scientific, Mapping):
            raise RuntimeError("role-neutral result lacks its scientific identity")
        value = scientific.get("content_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("role-neutral scientific identity is invalid")
        return value


@dataclass(frozen=True)
class _CompletedArtifactTarget:
    """One checkpoint-bound complete fit reopened by the terminal audit."""

    root: Path
    workload: RoleNeutralBenchmarkWorkload
    manifest_content_sha256: str
    scientific_artifact_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise ValueError("benchmark audit target root must be absolute")
        object.__setattr__(self, "root", root)
        if not isinstance(self.workload, RoleNeutralBenchmarkWorkload):
            raise TypeError("benchmark audit target requires a typed workload")
        for label, value in (
            ("manifest", self.manifest_content_sha256),
            ("scientific artifact", self.scientific_artifact_sha256),
        ):
            if _SHA256.fullmatch(str(value)) is None:
                raise ValueError(
                    f"benchmark audit target {label} identity is invalid"
                )


def _child_process_io_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[
    bool,
    int | None,
    int | None,
    float | None,
    float | None,
    dict[str, int] | None,
]:
    """Return one authenticated child-process delta for a one-owner fit.

    Process-isolated GPU telemetry is closed over every device reserved by the
    owner.  The returned peak map therefore preserves a spanning learned-query
    context instead of collapsing it onto the owner's primary device.
    """

    summary = manifest.get("owner_execution_telemetry")
    if not isinstance(summary, Mapping):
        raise ValueError("benchmark execution manifest lacks owner telemetry")
    process_isolated = summary.get("process_isolated_physical_owners")
    if type(process_isolated) is not bool:
        raise ValueError("benchmark owner telemetry lacks its execution kind")
    if process_isolated is False:
        return False, None, None, None, None, None
    if (
        summary.get("parent_process_counters_included_in_child_counters")
        is not False
    ):
        raise ValueError("child process counters include parent process bytes")
    owners = summary.get("physical_owners")
    if not isinstance(owners, list) or len(owners) != 1:
        raise ValueError(
            "one-owner benchmark execution has invalid child telemetry coverage"
        )
    owner_row = owners[0]
    telemetry = (
        owner_row.get("telemetry")
        if isinstance(owner_row, Mapping)
        else None
    )
    process_io = (
        telemetry.get("process_io_deltas")
        if isinstance(telemetry, Mapping)
        else None
    )
    required_io = {"rchar", "wchar", "read_bytes", "write_bytes"}
    wall = telemetry.get("wall_seconds") if isinstance(telemetry, Mapping) else None
    cpu = telemetry.get("cpu_seconds") if isinstance(telemetry, Mapping) else None
    peak_allocated = (
        telemetry.get("peak_gpu_allocated_bytes")
        if isinstance(telemetry, Mapping)
        else None
    )
    peak_reserved = (
        telemetry.get("peak_gpu_reserved_bytes")
        if isinstance(telemetry, Mapping)
        else None
    )
    if (
        not isinstance(telemetry, Mapping)
        or telemetry.get("schema_version")
        != "production_role_neutral_process_owner_telemetry_v1"
        or not isinstance(process_io, Mapping)
        or set(process_io) != required_io
        or any(
            isinstance(process_io[name], bool)
            or not isinstance(process_io[name], int)
            or process_io[name] < 0
            for name in required_io
        )
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) <= 0
        or isinstance(cpu, bool)
        or not isinstance(cpu, (int, float))
        or not math.isfinite(float(cpu))
        or float(cpu) < 0
        or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            )
            for value in (peak_allocated, peak_reserved)
        )
    ):
        raise ValueError(
            "process-isolated benchmark lacks closed child I/O counters"
        )
    owner_resource = (
        owner_row.get("resource")
        if isinstance(owner_row, Mapping)
        else None
    )
    telemetry_resource = (
        telemetry.get("resource")
        if isinstance(telemetry, Mapping)
        else None
    )
    reserved_resources = (
        telemetry.get("reserved_resources")
        if isinstance(telemetry, Mapping)
        else None
    )
    peak_allocated_by_device = (
        telemetry.get("peak_gpu_allocated_bytes_by_device")
        if isinstance(telemetry, Mapping)
        else None
    )
    peak_reserved_by_device = (
        telemetry.get("peak_gpu_reserved_bytes_by_device")
        if isinstance(telemetry, Mapping)
        else None
    )
    resources_are_closed = (
        isinstance(owner_resource, str)
        and isinstance(telemetry_resource, str)
        and telemetry_resource == owner_resource
        and isinstance(reserved_resources, list)
        and bool(reserved_resources)
        and all(
            isinstance(value, str) and bool(value)
            for value in reserved_resources
        )
        and len(reserved_resources) == len(set(reserved_resources))
        and reserved_resources[0] == owner_resource
    )
    reserved_gpu_devices = (
        set()
        if reserved_resources == ["cpu"]
        else (
            set(reserved_resources)
            if isinstance(reserved_resources, list)
            and all(
                isinstance(value, str)
                and value.startswith("cuda:")
                and value.split(":", 1)[1].isdigit()
                for value in reserved_resources
            )
            else None
        )
    )
    maps_are_closed = (
        resources_are_closed
        and reserved_gpu_devices is not None
        and isinstance(peak_allocated_by_device, Mapping)
        and isinstance(peak_reserved_by_device, Mapping)
        and set(peak_allocated_by_device) == reserved_gpu_devices
        and set(peak_reserved_by_device) == reserved_gpu_devices
        and all(
            not isinstance(value, bool)
            and isinstance(value, int)
            and value >= 0
            for mapping in (
                peak_allocated_by_device,
                peak_reserved_by_device,
            )
            for value in mapping.values()
        )
    )
    primary_peaks_are_consistent = (
        (
            reserved_gpu_devices == set()
            and peak_allocated is None
            and peak_reserved is None
        )
        or (
            isinstance(reserved_gpu_devices, set)
            and bool(reserved_gpu_devices)
            and owner_resource in reserved_gpu_devices
            and not isinstance(peak_allocated, bool)
            and isinstance(peak_allocated, int)
            and not isinstance(peak_reserved, bool)
            and isinstance(peak_reserved, int)
            and isinstance(peak_allocated_by_device, Mapping)
            and isinstance(peak_reserved_by_device, Mapping)
            and peak_allocated_by_device.get(owner_resource)
            == peak_allocated
            and peak_reserved_by_device.get(owner_resource)
            == peak_reserved
        )
    )
    if not maps_are_closed or not primary_peaks_are_consistent:
        raise ValueError(
            "process-isolated benchmark lacks closed per-device GPU peaks"
        )
    peak_by_device = {
        device: max(
            int(peak_allocated_by_device[device]),
            int(peak_reserved_by_device[device]),
        )
        for device in sorted(reserved_gpu_devices)
    }
    return (
        True,
        int(process_io["read_bytes"]),
        int(process_io["write_bytes"]),
        float(wall),
        float(cpu),
        peak_by_device,
    )


def _child_cpu_budget_attestation(
    *,
    manifest: Mapping[str, Any],
    candidate: RoleNeutralBenchmarkCandidate,
    per_fit_cpu_budget: int,
) -> Mapping[str, Any] | None:
    summary = manifest.get("owner_execution_telemetry")
    owners = (
        summary.get("physical_owners")
        if isinstance(summary, Mapping)
        else None
    )
    process_isolated = (
        summary.get("process_isolated_physical_owners")
        if isinstance(summary, Mapping)
        else None
    )
    if process_isolated is False:
        return None
    telemetry = (
        owners[0].get("telemetry")
        if isinstance(owners, list)
        and len(owners) == 1
        and isinstance(owners[0], Mapping)
        else None
    )
    native_threads = (
        telemetry.get("native_threads")
        if isinstance(telemetry, Mapping)
        else None
    )
    if (
        process_isolated is not True
        or isinstance(native_threads, bool)
        or not isinstance(native_threads, int)
        or native_threads < 1
    ):
        raise ValueError(
            "process-isolated benchmark lacks its child CPU budget"
        )
    if candidate.executor_mode == "fresh_per_fit":
        if native_threads != int(per_fit_cpu_budget):
            raise ValueError(
                "fresh benchmark child changed its per-fit CPU budget"
            )
        return {
            "executor_mode": "fresh_per_fit",
            "native_threads": native_threads,
            "per_fit_cpu_budget": int(per_fit_cpu_budget),
            "host_cpu_budget": int(candidate.host_cpu_budget),
            "maximum_simultaneous_fit_cpu_budget": (
                int(per_fit_cpu_budget) * candidate.total_concurrency
            ),
            "budget_product_within_host": (
                int(per_fit_cpu_budget) * candidate.total_concurrency
                <= candidate.host_cpu_budget
            ),
        }
    slot_budget = telemetry.get("slot_cpu_budget")
    host_budget = telemetry.get("host_cpu_budget")
    active_slots = telemetry.get("active_slot_count")
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in (slot_budget, host_budget, active_slots)
        )
        or int(host_budget) != candidate.host_cpu_budget
        or int(active_slots) != candidate.total_concurrency
        or int(slot_budget) != int(per_fit_cpu_budget)
        or native_threads != int(slot_budget)
        or int(slot_budget) * int(active_slots) > int(host_budget)
    ):
        raise ValueError(
            "persistent benchmark child CPU slot attestation is invalid"
        )
    return {
        "executor_mode": "persistent_slots",
        "native_threads": native_threads,
        "slot_cpu_budget": int(slot_budget),
        "host_cpu_budget": int(host_budget),
        "active_slot_count": int(active_slots),
        "maximum_simultaneous_fit_cpu_budget": (
            int(slot_budget) * int(active_slots)
        ),
        "budget_product_within_host": True,
    }


def _component_execution_intervals_from_manifest(
    *,
    manifest: Mapping[str, Any],
    expected_physical_owner_scope_id: str,
    expected_primary_resource: str,
    expected_neural_query_resources: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Reopen the directly measured component envelopes from one fit.

    The records time architecture-phase execution in the worker's monotonic
    clock domain.  Accelerator-associated envelopes include their host-side
    orchestration and are therefore suitable only for descriptive lane
    scheduling analysis, never kernel occupancy or causal speedup claims.
    """

    summary = manifest.get("owner_execution_telemetry")
    owners = (
        summary.get("physical_owners")
        if isinstance(summary, Mapping)
        else None
    )
    owner = (
        owners[0]
        if isinstance(owners, list)
        and len(owners) == 1
        and isinstance(owners[0], Mapping)
        else None
    )
    telemetry = (
        owner.get("telemetry")
        if isinstance(owner, Mapping)
        else None
    )
    if (
        not isinstance(owner, Mapping)
        or owner.get("physical_owner_scope_id")
        != expected_physical_owner_scope_id
    ):
        raise ValueError(
            "benchmark fit lacks complete component execution intervals"
        )
    return validate_role_neutral_component_execution_intervals(
        execution_telemetry=telemetry,
        expected_physical_owner_scope_id=(
            expected_physical_owner_scope_id
        ),
        expected_primary_resource=expected_primary_resource,
        expected_neural_query_resources=(
            expected_neural_query_resources
        ),
    )


def _fit_lane_intervals_from_component_execution(
    *,
    owner_execution_id: str,
    component_execution_intervals: Sequence[Mapping[str, Any]],
) -> tuple[FitLaneInterval, ...]:
    """Translate one already-validated six-component report losslessly."""

    rows = tuple(component_execution_intervals)
    if len(rows) != len(EXPECTED_COMPONENT_FAMILIES):
        raise ValueError(
            "completed accelerator fit lacks all component intervals"
        )
    translated: list[FitLaneInterval] = []
    for component, row in zip(
        EXPECTED_COMPONENT_FAMILIES,
        rows,
        strict=True,
    ):
        if not isinstance(row, Mapping) or row.get("component") != component:
            raise ValueError(
                "component interval order changed before lane translation"
            )
        translated.append(
            FitLaneInterval(
                interval_id=f"{owner_execution_id}.{component}",
                owner_execution_id=owner_execution_id,
                lane_kind=str(row["lane_kind"]),
                subphase_name=(
                    f"{row['lane_kind']}_associated_"
                    f"architecture_phase.{component}"
                ),
                resource_id="+".join(
                    str(value) for value in row["resource_ids"]
                ),
                clock_domain_id=str(row["clock_domain_id"]),
                started_monotonic_ns=int(
                    row["started_monotonic_ns"]
                ),
                finished_monotonic_ns=int(
                    row["finished_monotonic_ns"]
                ),
                status=str(row["status"]),
                timestamps_measured_directly=bool(
                    row["timestamps_measured_directly"]
                ),
            )
        )
    return tuple(translated)


def _htr_operational_attestation(
    *,
    manifest: Mapping[str, Any],
    candidate: RoleNeutralBenchmarkCandidate,
) -> Mapping[str, Any]:
    summary = manifest.get("owner_execution_telemetry")
    owners = (
        summary.get("physical_owners")
        if isinstance(summary, Mapping)
        else None
    )
    telemetry = (
        owners[0].get("telemetry")
        if isinstance(owners, list)
        and len(owners) == 1
        and isinstance(owners[0], Mapping)
        else None
    )
    if not isinstance(telemetry, Mapping):
        raise ValueError("benchmark lacks HTR owner operational telemetry")
    report = telemetry.get("worker_report", telemetry)
    if not isinstance(report, Mapping):
        raise ValueError("benchmark lacks its HTR worker report")
    component_reports = report.get("component_reports")
    attestation = (
        component_reports.get("htr")
        if isinstance(component_reports, Mapping)
        else None
    )
    if not isinstance(attestation, Mapping):
        raise ValueError("benchmark HTR worker report omitted its attestation")
    body = {
        key: copy.deepcopy(value)
        for key, value in attestation.items()
        if key != "content_sha256"
    }
    if (
        attestation.get("schema_version")
        != "production_role_neutral_htr_operational_attestation_v2"
        or attestation.get("content_sha256") != identity_sha256(body)
        or attestation.get("controls")
        != candidate.htr_operational_controls.as_dict()
        or attestation.get("training_batch_override_applied") is not False
        or attestation.get(
            "operational_predictions_within_declared_tolerance_of_scientific_replay"
        )
        is not True
        or attestation.get("replay_comparison_policy")
        != NEURAL_REPLAY_COMPARISON_POLICY
        or attestation.get("cache_capacities_nonbinding") is not True
        or attestation.get(
            "positive_data_loader_workers_exercised"
        )
        is not True
        or attestation.get("semantic_truncation_applied") is not False
    ):
        raise ValueError("benchmark HTR operational attestation changed")
    validate_neural_replay_settings(
        policy=attestation["replay_comparison_policy"],
        relative_tolerance=attestation.get("replay_relative_tolerance"),
        absolute_tolerance=attestation.get("replay_absolute_tolerance"),
    )
    reuse = (
        candidate.htr_operational_controls.reuse_tokenizer_and_chunk_plans
    )
    if reuse != (
        isinstance(attestation.get("fit_reusable_plan"), Mapping)
        and isinstance(
            attestation.get("exact_heldout_reusable_plan"),
            Mapping,
        )
    ):
        raise ValueError(
            "benchmark HTR reusable-plan attestation differs from controls"
        )
    reusable_plans = tuple(
        value
        for value in (
            attestation.get("fit_reusable_plan"),
            attestation.get("exact_heldout_reusable_plan"),
        )
        if isinstance(value, Mapping)
    )
    if reuse and any(
        int(value.get("unique_note_count", -1))
        > candidate.htr_operational_controls.chunk_plan_cache_max_entries
        or int(value.get("unique_chunk_count", -1))
        > candidate.htr_operational_controls.tokenized_chunk_cache_max_entries
        for value in reusable_plans
    ):
        raise ValueError(
            "benchmark HTR reusable-plan capacity bound complete evidence"
        )
    if (
        candidate.htr_operational_controls.data_loader_workers > 0
        and (
            len(reusable_plans) != 2
            or any(
                value.get("positive_data_loader_workers_exercised")
                is not True
                or int(value.get("parallel_plan_task_count", 0)) < 1
                or int(value.get("parallel_plan_thread_count", 0)) < 1
                for value in reusable_plans
            )
        )
    ):
        raise ValueError(
            "benchmark HTR positive data-loader workers lack executed work"
        )
    return copy.deepcopy(dict(attestation))


def _neural_query_topology_attestation(
    *,
    manifest: Mapping[str, Any],
    candidate: RoleNeutralBenchmarkCandidate,
    device: str,
    candidate_devices: tuple[str, ...],
) -> Mapping[str, Any] | None:
    summary = manifest.get("owner_execution_telemetry")
    if not isinstance(summary, Mapping):
        raise ValueError(
            "benchmark execution manifest lacks owner telemetry"
        )
    process_isolated = summary.get("process_isolated_physical_owners")
    if process_isolated is False:
        return None
    owners = summary.get("physical_owners")
    telemetry = (
        owners[0].get("telemetry")
        if isinstance(owners, list)
        and len(owners) == 1
        and isinstance(owners[0], Mapping)
        else None
    )
    attestation = (
        telemetry.get("neural_query_device_topology")
        if isinstance(telemetry, Mapping)
        else None
    )
    expected = candidate.neural_query_topology.runtime_topologies(
        candidate_devices
    )[device]
    if (
        process_isolated is not True
        or not isinstance(attestation, Mapping)
        or attestation.get("schema_version")
        != "neural_query_runtime_device_topology_attestation_v1"
        or attestation.get("devices") != list(expected.devices)
        or attestation.get("homogeneous") is not True
        or attestation.get("scientific_identity_includes_topology") is not False
        or (
            isinstance(telemetry, Mapping)
            and telemetry.get("reserved_resources")
            != list(expected.devices)
        )
    ):
        raise ValueError(
            "process-isolated benchmark did not execute its configured "
            "learned-query device topology"
        )
    return copy.deepcopy(dict(attestation))


class _CandidateGpuSampler:
    """Continuously sample all candidate devices across the whole observation."""

    def __init__(
        self,
        *,
        devices: tuple[str, ...],
        interval_seconds: float,
    ) -> None:
        self.devices = devices
        self.interval_seconds = _finite_positive(
            interval_seconds,
            label="GPU sample interval",
        )
        self._samples: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        sampled_at = time.monotonic()
        rows = telemetry_module.sample_nvidia_gpus(self.devices)
        with self._lock:
            self._samples.extend(
                {
                    **dict(value),
                    "sample_monotonic_seconds": sampled_at,
                }
                for value in rows
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> "_CandidateGpuSampler":
        if self.devices != ("cpu",):
            self._sample()
            self._thread = threading.Thread(
                target=self._run,
                name="oci-role-neutral-gpu-sampler",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._sample()

    @property
    def samples(self) -> tuple[Mapping[str, Any], ...]:
        with self._lock:
            return tuple(copy.deepcopy(self._samples))


def _instrumented_factories(
    factories: RoleNeutralProducerFactories,
    *,
    ledger: TelemetryLedger,
    scope_label: str,
    instance_index: int,
) -> RoleNeutralProducerFactories:
    originals = factories.as_mapping()
    wrapped: dict[str, Callable[..., BoundRoleNeutralComponentProducer]] = {}
    for component in EXPECTED_COMPONENT_FAMILIES:
        original = originals[component]

        def factory(invocation: Any, *, _component: str = component, _original: Any = original):
            bound = _original(invocation)
            if not isinstance(bound, BoundRoleNeutralComponentProducer):
                raise TypeError("benchmark producer factory returned an untyped binding")

            def execute() -> Any:
                with ledger.subphase(
                    f"fit.{instance_index}.{_component}",
                    activity_kind="model_fit",
                    scope_label=scope_label,
                ):
                    return bound.execute()

            return BoundRoleNeutralComponentProducer(
                execute=execute,
                authenticate=bound.authenticate,
            )

        wrapped[component] = factory
    return RoleNeutralProducerFactories(**wrapped)


def _is_oom(exc: BaseException) -> bool:
    if "outofmemory" in type(exc).__name__.lower():
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def _memory_observation(
    *,
    devices: tuple[str, ...],
    inventory: ResourceInventory,
    complete_record: SubphaseTelemetry,
) -> tuple[float | None, int | None, bool]:
    if devices == ("cpu",):
        return None, None, True
    by_device = {gpu.device: gpu for gpu in inventory.gpus}
    samples_by_device: dict[str, list[Mapping[str, Any]]] = {
        device: [] for device in devices
    }
    for sample in complete_record.gpu_samples:
        device = str(sample.get("device"))
        if device in samples_by_device:
            samples_by_device[device].append(sample)
    peaks: list[float] = []
    headrooms: list[int] = []
    complete = True
    for device in devices:
        gpu = by_device.get(device)
        samples = samples_by_device[device]
        if gpu is None or not samples:
            complete = False
            if gpu is None:
                continue
            used_values = [int(gpu.used_memory_bytes)]
            total = int(gpu.total_memory_bytes)
        else:
            total_values = {
                int(sample.get("memory_total_bytes", 0)) for sample in samples
            }
            if len(total_values) != 1 or next(iter(total_values)) <= 0:
                complete = False
                total = int(gpu.total_memory_bytes)
            else:
                total = next(iter(total_values))
            used_values = [
                int(sample.get("memory_used_bytes", 0)) for sample in samples
            ]
        torch_peak = int(complete_record.gpu_peak_memory_bytes.get(device, 0))
        peak_used = max((*used_values, torch_peak))
        peaks.append(peak_used / total)
        headrooms.append(max(0, total - peak_used))
    if not peaks or not headrooms:
        return 1.0, 0, False
    return max(peaks), min(headrooms), complete


def _candidate_memory_observation(
    *,
    devices: tuple[str, ...],
    inventory: ResourceInventory,
    samples: Sequence[Mapping[str, Any]],
    instances: Sequence[_InstanceResult],
    concurrency_per_device: int,
) -> tuple[float | None, int | None, bool, dict[str, int]]:
    """Summarize continuously sampled total allocation plus Torch process peaks."""

    if devices == ("cpu",):
        return None, None, True, {}
    inventory_by_device = {gpu.device: gpu for gpu in inventory.gpus}
    samples_by_device = {
        device: [
            value
            for value in samples
            if str(value.get("device")) == device
        ]
        for device in devices
    }
    peak_fractions: list[float] = []
    headrooms: list[int] = []
    peak_bytes: dict[str, int] = {}
    complete = True
    for device in devices:
        gpu = inventory_by_device.get(device)
        rows = samples_by_device[device]
        if gpu is None or not rows:
            complete = False
            if gpu is None:
                continue
            total = int(gpu.total_memory_bytes)
            used = int(gpu.used_memory_bytes)
        else:
            totals = {
                int(value.get("memory_total_bytes", 0)) for value in rows
            }
            if len(totals) != 1 or next(iter(totals)) <= 0:
                complete = False
                total = int(gpu.total_memory_bytes)
            else:
                total = next(iter(totals))
            used = max(
                int(value.get("memory_used_bytes", 0)) for value in rows
            )
        parent_torch_peak = max(
            (
                int(
                    instance.complete_record.gpu_peak_memory_bytes.get(
                        device,
                        0,
                    )
                )
                for instance in instances
            ),
            default=0,
        )
        child_peaks = sorted(
            (
                int(instance.child_peak_gpu_memory_bytes_by_device[device])
                for instance in instances
                if (
                    instance.child_peak_gpu_memory_bytes_by_device is not None
                    and device
                    in instance.child_peak_gpu_memory_bytes_by_device
                )
            ),
            reverse=True,
        )
        concurrent_child_peak = sum(
            child_peaks[: int(concurrency_per_device)]
        )
        peak = max(
            used,
            parent_torch_peak,
            int(gpu.used_memory_bytes) + concurrent_child_peak,
        )
        peak_bytes[device] = peak
        peak_fractions.append(peak / total)
        headrooms.append(max(0, total - peak))
    if not peak_fractions or not headrooms:
        return 1.0, 0, False, peak_bytes
    return max(peak_fractions), min(headrooms), complete, peak_bytes


def _run_instance(
    *,
    instance_index: int,
    root: Path,
    workload: RoleNeutralBenchmarkWorkload,
    scope_label: str,
    device: str,
    candidate_devices: tuple[str, ...],
    candidate: RoleNeutralBenchmarkCandidate,
    per_fit_cpu_budget: int,
    physical_owner_executor: Any,
    inventory: ResourceInventory,
    safety: ResourcePerformanceSafetyPolicy,
) -> _InstanceResult:
    model_ledger = TelemetryLedger()
    topology_mapping = candidate.neural_query_topology.runtime_topologies(
        candidate_devices
    )
    topology = topology_mapping[device]
    complete_ledger = TelemetryLedger(devices=topology.devices)
    manifest: Mapping[str, Any] | None = None
    error: BaseException | None = None
    with complete_ledger.subphase(
        f"complete.{instance_index}",
        activity_kind="coordination_proof",
        scope_label=scope_label,
    ):
        try:
            factories = workload.producer_factories_builder()
            if not isinstance(factories, RoleNeutralProducerFactories):
                raise TypeError(
                    "benchmark workload returned untyped producer factories"
                )
            instrumented = _instrumented_factories(
                factories,
                ledger=model_ledger,
                scope_label=scope_label,
                instance_index=instance_index,
            )
            ordered_candidate_devices = (
                device,
                *(
                    candidate_device
                    for candidate_device in candidate_devices
                    if candidate_device != device
                ),
            )
            resource_plan = ResourcePlan(
                devices=ordered_candidate_devices,
                cpu_budget=(
                    int(candidate.host_cpu_budget)
                    if candidate.executor_mode == "persistent_slots"
                    else int(per_fit_cpu_budget)
                ),
                inventory=inventory,
                policy=candidate_devices,
                resource_performance_safety=safety,
            )
            manifest = execute_and_publish_role_neutral_stage1(
                root=root,
                plan=workload.plan,
                producer_factories=instrumented,
                policy=RoleNeutralStage1ExecutionPolicy(
                    resource_plan=resource_plan,
                    max_parallel_owners=1,
                    neural_query_execution_topologies={
                        primary: topology_mapping[primary]
                        for primary in ordered_candidate_devices
                    },
                    htr_operational_controls=(
                        candidate.htr_operational_controls
                    ),
                ),
                executor=physical_owner_executor,
            )
        except BaseException as exc:
            error = exc
            if not _is_oom(exc):
                raise
    complete_records = complete_ledger.records
    if len(complete_records) != 1:
        raise RuntimeError("benchmark complete-execution telemetry is incomplete")
    complete_record = complete_records[0]
    model_records = model_ledger.records
    (
        child_process_counters_expected,
        child_process_read_bytes,
        child_process_written_bytes,
        child_process_wall_seconds,
        child_process_cpu_seconds,
        child_peak_gpu_memory_bytes_by_device,
    ) = (
        (False, None, None, None, None, None)
        if manifest is None
        else _child_process_io_from_manifest(manifest)
    )
    child_cpu_budget_attestation = (
        None
        if manifest is None
        else _child_cpu_budget_attestation(
            manifest=manifest,
            candidate=candidate,
            per_fit_cpu_budget=per_fit_cpu_budget,
        )
    )
    htr_operational_attestation = (
        None
        if manifest is None
        else _htr_operational_attestation(
            manifest=manifest,
            candidate=candidate,
        )
    )
    neural_query_topology_attestation = (
        None
        if manifest is None
        else _neural_query_topology_attestation(
            manifest=manifest,
            candidate=candidate,
            device=device,
            candidate_devices=candidate_devices,
        )
    )
    component_execution_intervals = (
        ()
        if manifest is None
        else _component_execution_intervals_from_manifest(
            manifest=manifest,
            expected_physical_owner_scope_id=(
                workload.plan.physical_scopes[0].scope_id
            ),
            expected_primary_resource=device,
            expected_neural_query_resources=topology.devices,
        )
    )
    model_wall = (
        float(child_process_wall_seconds)
        if child_process_wall_seconds is not None
        else sum(value.wall_seconds for value in model_records)
    )
    model_cpu = (
        float(child_process_cpu_seconds)
        if child_process_cpu_seconds is not None
        else sum(value.cpu_seconds for value in model_records)
    )
    coordination_wall = max(0.0, complete_record.wall_seconds - model_wall)
    coordination_cpu = max(0.0, complete_record.cpu_seconds - model_cpu)
    logical_reads = int(complete_record.byte_counters.get("read", 0))
    peak, headroom, gpu_complete = _memory_observation(
        devices=topology.devices,
        inventory=inventory,
        complete_record=complete_record,
    )
    return _InstanceResult(
        instance_index=instance_index,
        root=root,
        device=device,
        manifest=manifest,
        model_records=model_records,
        complete_record=complete_record,
        model_wall_seconds=model_wall,
        model_cpu_seconds=model_cpu,
        coordination_wall_seconds=coordination_wall,
        coordination_cpu_seconds=coordination_cpu,
        ordinary_process_read_bytes=int(
            complete_record.process_read_bytes or 0
        ),
        ordinary_logical_read_bytes=logical_reads,
        child_process_counters_expected=child_process_counters_expected,
        child_process_read_bytes=child_process_read_bytes,
        child_process_written_bytes=child_process_written_bytes,
        child_process_wall_seconds=child_process_wall_seconds,
        child_process_cpu_seconds=child_process_cpu_seconds,
        child_peak_gpu_memory_bytes_by_device=(
            child_peak_gpu_memory_bytes_by_device
        ),
        child_cpu_budget_attestation=child_cpu_budget_attestation,
        htr_operational_attestation=htr_operational_attestation,
        neural_query_topology_attestation=(
            neural_query_topology_attestation
        ),
        component_execution_intervals=(
            component_execution_intervals
        ),
        peak_allocation_fraction=peak,
        minimum_headroom_bytes=headroom,
        gpu_telemetry_complete=gpu_complete,
        oom_observed=bool(error is not None),
    )


def _safe_gpu_pool(
    *,
    inventory: ResourceInventory,
    safety: ResourcePerformanceSafetyPolicy,
) -> tuple[GPUResource, ...]:
    values: list[GPUResource] = []
    for gpu in inventory.gpus:
        if (
            safety.fail_on_external_gpu_occupants
            and gpu.external_processes
        ):
            continue
        if gpu.free_memory_bytes < safety.gpu_minimum_headroom_bytes:
            continue
        if (
            gpu.used_memory_bytes / gpu.total_memory_bytes
            >= safety.gpu_max_allocation_fraction
        ):
            continue
        values.append(gpu)
    return tuple(values)


def _candidate_devices(
    *,
    candidate: RoleNeutralBenchmarkCandidate,
    inventory: ResourceInventory,
    safety: ResourcePerformanceSafetyPolicy,
) -> tuple[str, ...]:
    if candidate.accelerator_count == 0:
        return ("cpu",)
    pool = _safe_gpu_pool(inventory=inventory, safety=safety)
    if len(pool) < candidate.accelerator_count:
        raise RuntimeError(
            f"candidate {candidate.name!r} requires "
            f"{candidate.accelerator_count} safe accelerators but only "
            f"{len(pool)} are available"
        )
    return tuple(
        value.device for value in pool[: candidate.accelerator_count]
    )


def _phase_rows(instance: _InstanceResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in instance.model_records:
        row = value.as_dict()
        # These are process-global diagnostic spans and may overlap peer fits.
        # The observation-wide counter below is the only additive authority.
        row["process_read_bytes"] = None
        row["process_written_bytes"] = None
        row["process_counter_attribution"] = (
            "diagnostic_span_not_additive"
        )
        rows.append(row)
    if instance.child_process_counters_expected:
        rows.append(
            {
                "name": f"model_fit.child_process.{instance.instance_index}",
                "activity_kind": "model_fit",
                "scope_label": instance.complete_record.scope_label,
                "wall_seconds": instance.child_process_wall_seconds,
                "cpu_seconds": instance.child_process_cpu_seconds,
                "process_read_bytes": instance.child_process_read_bytes,
                "process_written_bytes": (
                    instance.child_process_written_bytes
                ),
                "process_counter_attribution": (
                    "one_isolated_child_delta_for_this_fit"
                ),
                "cpu_budget_attestation": (
                    None
                    if instance.child_cpu_budget_attestation is None
                    else dict(instance.child_cpu_budget_attestation)
                ),
                "byte_counters": {
                    key: 0
                    for key in instance.complete_record.byte_counters
                },
                "gpu_samples": [],
                "gpu_peak_memory_bytes": {},
                "status": instance.complete_record.status,
            }
        )
    complete = instance.complete_record
    coordination_counters = dict(complete.byte_counters)
    for model in instance.model_records:
        for key, value in model.byte_counters.items():
            coordination_counters[key] = max(
                0,
                int(coordination_counters.get(key, 0)) - int(value),
            )
    rows.append(
        {
            "name": f"coordination_proof.{instance.instance_index}",
            "activity_kind": "coordination_proof",
            "scope_label": complete.scope_label,
            "wall_seconds": instance.coordination_wall_seconds,
            "cpu_seconds": instance.coordination_cpu_seconds,
            "process_read_bytes": None,
            "process_written_bytes": None,
            "process_counter_attribution": (
                "diagnostic_span_not_additive"
            ),
            "byte_counters": coordination_counters,
            "gpu_samples": [
                dict(value) for value in complete.gpu_samples
            ],
            "gpu_peak_memory_bytes": dict(complete.gpu_peak_memory_bytes),
            "status": complete.status,
        }
    )
    return rows


def _observation_process_io() -> tuple[int, int] | None:
    """Sample process I/O once around a whole concurrency observation.

    Per-fit ``/proc/self/io`` spans overlap when fits share an interpreter, so
    summing them amplifies the same process-global bytes.  Keeping this seam
    separate also lets process-isolated executors supply child counters without
    mixing them with the parent observation.
    """

    return telemetry_module._proc_io()


def _run_observation(
    *,
    root: Path,
    config: RoleNeutralBenchmarkConfig,
    candidate: RoleNeutralBenchmarkCandidate,
    scope: RoleNeutralBenchmarkScope,
    workload: RoleNeutralBenchmarkWorkload,
    repetition_index: int,
    inventory: ResourceInventory,
    physical_owner_executor: Any,
    observation_kind: str = "measured",
) -> tuple[
    BenchmarkRunObservation,
    dict[str, Any],
    tuple[_InstanceResult, ...],
]:
    devices = _candidate_devices(
        candidate=candidate,
        inventory=inventory,
        safety=config.resource_performance_safety,
    )
    slots = tuple(
        device
        for device in devices
        for _index in range(candidate.concurrency_per_device)
    )
    per_fit_cpu_budget = max(
        1,
        int(candidate.host_cpu_budget) // int(candidate.total_concurrency),
    )
    if per_fit_cpu_budget * candidate.total_concurrency > candidate.host_cpu_budget:
        raise RuntimeError(
            "derived per-fit CPU allocation exceeds the candidate host budget"
        )
    if observation_kind not in {"warmup", "measured"}:
        raise ValueError("benchmark observation_kind is unsupported")
    observation_root = (
        root
        / ("warmups" if observation_kind == "warmup" else "runs")
        / candidate.name
        / scope.label
        / (
            f"warmup_{repetition_index:03d}"
            if observation_kind == "warmup"
            else f"repetition_{repetition_index:03d}"
        )
    )
    observation_root.mkdir(parents=True, exist_ok=False)
    process_io_before = _observation_process_io()
    observation_started_monotonic_ns = time.monotonic_ns()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    results: list[_InstanceResult] = []
    gpu_sampler = _CandidateGpuSampler(
        devices=devices,
        interval_seconds=config.gpu_sample_interval_seconds,
    )
    with gpu_sampler:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=candidate.total_concurrency,
            thread_name_prefix="oci-role-neutral-benchmark",
        ) as pool:
            futures = {
                pool.submit(
                    _run_instance,
                    instance_index=index,
                    root=(observation_root / f"fit_{index:03d}").resolve(),
                    workload=workload,
                    scope_label=scope.label,
                    device=slots[index % len(slots)],
                    candidate_devices=devices,
                    candidate=candidate,
                    per_fit_cpu_budget=per_fit_cpu_budget,
                    physical_owner_executor=physical_owner_executor,
                    inventory=inventory,
                    safety=config.resource_performance_safety,
                ): index
                for index in range(scope.fits_per_observation)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    observation_finished_monotonic_ns = time.monotonic_ns()
    batch_wall = time.perf_counter() - start_wall
    batch_cpu = time.process_time() - start_cpu
    process_io_after = _observation_process_io()
    observation_process_read_bytes = (
        None
        if process_io_before is None or process_io_after is None
        else max(0, int(process_io_after[0]) - int(process_io_before[0]))
    )
    observation_process_written_bytes = (
        None
        if process_io_before is None or process_io_after is None
        else max(0, int(process_io_after[1]) - int(process_io_before[1]))
    )
    ordered = tuple(sorted(results, key=lambda value: value.instance_index))
    completed = tuple(value for value in ordered if value.manifest is not None)
    oom_observed = any(value.oom_observed for value in ordered)
    hashes = {
        str(value.scientific_artifact_sha256)
        for value in completed
        if value.scientific_artifact_sha256 is not None
    }
    complete_equal = (
        len(completed) == scope.fits_per_observation and len(hashes) == 1
    )
    artifact_sha256 = (
        next(iter(hashes))
        if complete_equal and not oom_observed
        else None
    )
    lane_interval_observation: CompletedFitIntervalObservation | None = None
    lane_overlap_analysis = None
    lane_telemetry_required = devices != ("cpu",)
    if lane_telemetry_required and len(completed) == scope.fits_per_observation:
        fit_intervals: list[FitLaneInterval] = []
        owner_execution_ids: list[str] = []
        for instance in ordered:
            owner_execution_id = f"fit_{instance.instance_index:03d}"
            owner_execution_ids.append(owner_execution_id)
            fit_intervals.extend(
                _fit_lane_intervals_from_component_execution(
                    owner_execution_id=owner_execution_id,
                    component_execution_intervals=(
                        instance.component_execution_intervals
                    ),
                )
            )
        lane_interval_observation = (
            CompletedFitIntervalObservation.seal(
                observation_id=(
                    f"{observation_kind}:{candidate.name}:"
                    f"{scope.label}:{repetition_index}"
                ),
                owner_execution_ids=owner_execution_ids,
                clock_domain_id=(
                    ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN
                ),
                observation_started_monotonic_ns=(
                    observation_started_monotonic_ns
                ),
                observation_finished_monotonic_ns=(
                    observation_finished_monotonic_ns
                ),
                intervals=fit_intervals,
            )
        )
        lane_overlap_analysis = analyze_completed_fit_lane_overlap(
            lane_interval_observation,
            expected_observation_id=(
                lane_interval_observation.observation_id
            ),
            expected_owner_execution_ids=tuple(
                owner_execution_ids
            ),
        )
    lane_telemetry_complete = (
        not lane_telemetry_required
        or (
            lane_interval_observation is not None
            and lane_overlap_analysis is not None
        )
    )
    (
        peak,
        headroom,
        gpu_complete,
        gpu_peak_memory_bytes,
    ) = _candidate_memory_observation(
        devices=devices,
        inventory=inventory,
        samples=gpu_sampler.samples,
        instances=ordered,
        concurrency_per_device=candidate.concurrency_per_device,
    )
    model_wall = sum(value.model_wall_seconds for value in ordered)
    model_cpu = sum(value.model_cpu_seconds for value in ordered)
    coordination_wall = sum(
        value.coordination_wall_seconds for value in ordered
    )
    coordination_cpu = sum(
        value.coordination_cpu_seconds for value in ordered
    )
    read_source = config.resource_performance_safety.read_counter_source
    process_isolated_observation = any(
        value.child_process_counters_expected for value in ordered
    )
    if process_isolated_observation:
        process_counter_complete = (
            len(ordered) == scope.fits_per_observation
            and all(
                value.child_process_counters_expected
                and value.child_process_read_bytes is not None
                and value.child_process_written_bytes is not None
                for value in ordered
            )
        )
        attributed_process_read_bytes = sum(
            int(value.child_process_read_bytes or 0) for value in ordered
        )
        attributed_process_written_bytes: int | None = sum(
            int(value.child_process_written_bytes or 0) for value in ordered
        )
        process_counter_attribution = (
            "one_isolated_child_delta_per_complete_fit"
        )
    else:
        process_counter_complete = observation_process_read_bytes is not None
        attributed_process_read_bytes = int(
            observation_process_read_bytes or 0
        )
        attributed_process_written_bytes = (
            None
            if observation_process_written_bytes is None
            else int(observation_process_written_bytes)
        )
        process_counter_attribution = (
            "one_parent_process_delta_per_complete_observation"
        )
    counter_complete = (
        process_counter_complete
        if read_source == "process_read_bytes"
        else True
    )
    ordinary_read_bytes = (
        attributed_process_read_bytes
        if read_source == "process_read_bytes"
        else sum(value.ordinary_logical_read_bytes for value in ordered)
    )
    input_opportunity_bytes = (
        workload.unique_immutable_input_bytes * scope.fits_per_observation
    )
    overhead_ratio = (
        None if model_wall <= 0 else coordination_wall / model_wall
    )
    read_ratio = ordinary_read_bytes / input_opportunity_bytes
    memory_accepted = (
        devices == ("cpu",)
        or (
            gpu_complete
            and peak is not None
            and peak < config.resource_performance_safety.gpu_max_allocation_fraction
            and headroom is not None
            and headroom
            >= config.resource_performance_safety.gpu_minimum_headroom_bytes
        )
    )
    telemetry_accepted = (
        overhead_ratio is not None
        and overhead_ratio
        <= config.resource_performance_safety.maximum_coordination_proof_overhead_ratio
        and read_ratio
        <= config.resource_performance_safety.maximum_ordinary_read_amplification
        and gpu_complete
        and memory_accepted
        and counter_complete
        and lane_telemetry_complete
        and not oom_observed
        and complete_equal
    )
    observation = BenchmarkRunObservation(
        candidate_name=candidate.name,
        scope_label=scope.label,
        repetition_index=repetition_index,
        device_ids=devices,
        concurrency_per_device=candidate.concurrency_per_device,
        completed_scope_fits=len(completed),
        model_fit_wall_seconds=max(model_wall, batch_wall),
        end_to_end_wall_seconds=batch_wall,
        peak_allocation_fraction=peak,
        minimum_observed_headroom_bytes=headroom,
        oom_observed=oom_observed,
        scientific_artifact_sha256=artifact_sha256,
        artifact_path=str(observation_root),
        complete_artifacts_exactly_equal=complete_equal,
    )
    detail = {
        "candidate_name": candidate.name,
        "executor_mode": candidate.executor_mode,
        "observation_kind": observation_kind,
        "scope_label": scope.label,
        "repetition_index": repetition_index,
        "configured_fit_row_count": scope.fit_row_count,
        "configured_fits_per_observation": scope.fits_per_observation,
        "device_ids": list(devices),
        "concurrency_per_device": candidate.concurrency_per_device,
        "resource_slot_count": candidate.resource_slot_count,
        "effective_parallel_owners": candidate.total_concurrency,
        "neural_query_topology": (
            candidate.neural_query_topology.as_dict()
        ),
        "host_cpu_budget": candidate.host_cpu_budget,
        "per_fit_cpu_budget": per_fit_cpu_budget,
        "maximum_simultaneous_fit_cpu_budget": (
            per_fit_cpu_budget * candidate.total_concurrency
        ),
        "child_cpu_budget_attestations": [
            {
                "fit_index": value.instance_index,
                "attestation": (
                    None
                    if value.child_cpu_budget_attestation is None
                    else dict(value.child_cpu_budget_attestation)
                ),
            }
            for value in ordered
        ],
        "htr_operational_attestations": [
            {
                "fit_index": value.instance_index,
                "attestation": (
                    None
                    if value.htr_operational_attestation is None
                    else dict(value.htr_operational_attestation)
                ),
            }
            for value in ordered
        ],
        "neural_query_topology_attestations": [
            {
                "fit_index": value.instance_index,
                "attestation": (
                    None
                    if value.neural_query_topology_attestation is None
                    else dict(
                        value.neural_query_topology_attestation
                    )
                ),
            }
            for value in ordered
        ],
        "batch_wall_seconds": batch_wall,
        "batch_cpu_seconds": batch_cpu,
        "model_fit_wall_seconds_sum": model_wall,
        "model_fit_cpu_seconds_sum": model_cpu,
        "coordination_proof_wall_seconds_sum": coordination_wall,
        "coordination_proof_cpu_seconds_sum": coordination_cpu,
        "coordination_proof_overhead_ratio": overhead_ratio,
        "coordination_proof_overhead_limit": (
            config.resource_performance_safety.maximum_coordination_proof_overhead_ratio
        ),
        "ordinary_read_counter_source": read_source,
        "ordinary_read_counter_complete": counter_complete,
        "process_counter_attribution": (
            process_counter_attribution
            if read_source == "process_read_bytes"
            else "explicit_logical_byte_counters_per_fit"
        ),
        "observation_parent_process_read_bytes": (
            observation_process_read_bytes
        ),
        "observation_parent_process_written_bytes": (
            observation_process_written_bytes
        ),
        "attributed_process_read_bytes": attributed_process_read_bytes,
        "attributed_process_written_bytes": attributed_process_written_bytes,
        "ordinary_read_bytes": ordinary_read_bytes,
        "unique_input_read_opportunity_bytes": input_opportunity_bytes,
        "ordinary_read_amplification": read_ratio,
        "ordinary_read_amplification_limit": (
            config.resource_performance_safety.maximum_ordinary_read_amplification
        ),
        "terminal_audit_read_bytes_included": False,
        "peak_allocation_fraction": peak,
        "maximum_peak_allocation_fraction": (
            config.resource_performance_safety.gpu_max_allocation_fraction
        ),
        "minimum_observed_headroom_bytes": headroom,
        "minimum_required_headroom_bytes": (
            config.resource_performance_safety.gpu_minimum_headroom_bytes
        ),
        "gpu_telemetry_complete": gpu_complete,
        "gpu_sample_interval_seconds": config.gpu_sample_interval_seconds,
        "gpu_samples": [dict(value) for value in gpu_sampler.samples],
        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
        "gpu_peak_utilization_percent": (
            None
            if not gpu_sampler.samples
            else max(
                float(value.get("utilization_percent", 0.0))
                for value in gpu_sampler.samples
            )
        ),
        "cpu_gpu_lane_interval_telemetry_required": (
            lane_telemetry_required
        ),
        "cpu_gpu_lane_interval_telemetry_complete": (
            lane_telemetry_complete
        ),
        "cpu_gpu_lane_interval_semantics": (
            "direct_monotonic_architecture_phase_envelopes_"
            "not_kernel_occupancy_v1"
        ),
        "cpu_gpu_lane_interval_observation": (
            None
            if lane_interval_observation is None
            else lane_interval_observation.as_dict()
        ),
        "cpu_gpu_lane_overlap_analysis": (
            None
            if lane_overlap_analysis is None
            else lane_overlap_analysis.as_dict()
        ),
        "cpu_gpu_lane_overlap_descriptive_only": (
            lane_overlap_analysis is not None
            and lane_overlap_analysis.descriptive_overlap_only
        ),
        "cpu_gpu_lane_overlap_speedup_claimed": False,
        "oom_observed": oom_observed,
        "complete_scientific_artifacts_exactly_equal": complete_equal,
        "scientific_artifact_sha256": artifact_sha256,
        "phase_telemetry": [
            row for value in ordered for row in _phase_rows(value)
        ],
        "telemetry_accepted": telemetry_accepted,
    }
    return observation, detail, ordered


def _audit_tree(
    root: Path,
    *,
    ledger: TelemetryLedger,
) -> tuple[str, int, int]:
    inventory: list[dict[str, Any]] = []
    total = 0
    file_count = 0
    for path in sorted(
        root.rglob("*"),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("terminal benchmark audit encountered a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ValueError("terminal benchmark audit encountered non-private data")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                size += len(block)
        if size != int(metadata.st_size):
            raise RuntimeError("terminal benchmark audit observed a changing file")
        total += size
        file_count += 1
        ledger.count_bytes(read=size, hashed=size)
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return identity_sha256(inventory), total, file_count


def _terminal_audit(
    *,
    completed: Sequence[_CompletedArtifactTarget],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = TelemetryLedger()
    rows: list[dict[str, Any]] = []
    with ledger.subphase(
        "benchmark.terminal_complete_artifact_audit",
        activity_kind="terminal_audit",
    ):
        for target in completed:
            validated = validate_role_neutral_stage1_execution(
                root=target.root,
                plan=target.workload.plan,
            )
            if (
                validated.get("content_sha256")
                != target.manifest_content_sha256
                or (
                    validated.get("scientific_identity", {}).get(
                        "content_sha256"
                    )
                    if isinstance(
                        validated.get("scientific_identity"),
                        Mapping,
                    )
                    else None
                )
                != target.scientific_artifact_sha256
            ):
                raise RuntimeError(
                    "terminal audit found a changed role-neutral execution"
                )
            tree_sha256, total_bytes, file_count = _audit_tree(
                target.root,
                ledger=ledger,
            )
            rows.append(
                {
                    "root": str(target.root),
                    "tree_sha256": tree_sha256,
                    "total_file_bytes": total_bytes,
                    "file_count": file_count,
                    "scientific_artifact_sha256": (
                        target.scientific_artifact_sha256
                    ),
                }
            )
    records = ledger.as_dict()["subphases"]
    if len(records) != 1 or records[0]["status"] != "completed":
        raise RuntimeError("terminal benchmark audit did not complete exactly once")
    return (
        {
            "exactly_one_completed_terminal_audit": True,
            "audited_complete_artifact_count": len(rows),
            "artifacts": rows,
        },
        ledger.as_dict(),
    )


def _write_result(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("benchmark result write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_closed_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read one private immutable JSON authority with a stable-byte check."""

    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be one private regular file")
    source = path.resolve(strict=True)
    if source != Path(os.path.abspath(os.fspath(path))):
        raise ValueError(f"{label} path must be canonical and symlink-free")
    payload = source.read_bytes()
    after = os.lstat(source)
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"{label} changed while it was being read")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not closed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _benchmark_execution_schedule(
    config: RoleNeutralBenchmarkConfig,
) -> dict[str, Any]:
    candidates = tuple(config.candidates)
    entries: list[dict[str, Any]] = []
    sequence_index = 0
    repetitions = (
        config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
    )
    for scope_index, scope in enumerate(config.representative_scopes):
        for warmup_index in range(
            config.warmup_observations_per_candidate_scope
        ):
            rotation_offset = (
                scope_index
                * config.warmup_observations_per_candidate_scope
                + warmup_index
            ) % len(candidates)
            rotated = (
                candidates[rotation_offset:]
                + candidates[:rotation_offset]
            )
            for candidate_position, candidate in enumerate(rotated):
                entries.append(
                    {
                        "sequence_index": sequence_index,
                        "observation_kind": "warmup",
                        "scope_label": scope.label,
                        "observation_index": warmup_index,
                        "rotation_offset": rotation_offset,
                        "candidate_position": candidate_position,
                        "candidate_name": candidate.name,
                    }
                )
                sequence_index += 1
        for repetition_index in range(repetitions):
            rotation_offset = (
                scope_index * repetitions + repetition_index
            ) % len(candidates)
            rotated = (
                candidates[rotation_offset:]
                + candidates[:rotation_offset]
            )
            for candidate_position, candidate in enumerate(rotated):
                entries.append(
                    {
                        "sequence_index": sequence_index,
                        "observation_kind": "measured",
                        "scope_label": scope.label,
                        "observation_index": repetition_index,
                        "rotation_offset": rotation_offset,
                        "candidate_position": candidate_position,
                        "candidate_name": candidate.name,
                    }
                )
                sequence_index += 1
    body = {
        "schema_version": (
            ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA
        ),
        "warmup_policy": (
            "configured_complete_observations_excluded_from_selection_v1"
        ),
        "warmup_observations_per_candidate_scope": (
            config.warmup_observations_per_candidate_scope
        ),
        "candidate_order_policy": (
            "scope_observation_latin_rotation_with_warmup_v2"
        ),
        "candidate_names_in_configured_order": [
            candidate.name for candidate in candidates
        ],
        "entries": entries,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _resource_resume_compatibility(
    resources: ResourceInventory,
) -> dict[str, Any]:
    """Stable hardware axes required to avoid mixing benchmark machines."""

    body = {
        "cpu_count": int(resources.cpu_count),
        "accelerators": [
            {
                "device": gpu.device,
                "uuid": gpu.uuid,
                "total_memory_bytes": int(gpu.total_memory_bytes),
            }
            for gpu in resources.gpus
        ],
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _benchmark_request(
    *,
    config: RoleNeutralBenchmarkConfig,
    workload_binding: Mapping[str, Any],
    typed_workloads: Mapping[str, RoleNeutralBenchmarkWorkload],
    compression_source: PortableProductionStage1ClusterPreflightArtifact,
    resources: ResourceInventory,
    execution_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    compression_identity = compression_source.identity()
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA,
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "workload_binding": copy.deepcopy(dict(workload_binding)),
        "immutable_inputs_by_scope": [
            {
                "scope_label": scope.label,
                "scientific_htr_training_batch_size": (
                    typed_workloads[
                        scope.label
                    ].scientific_htr_training_batch_size
                ),
                "inputs": [
                    asdict(value)
                    for value in typed_workloads[
                        scope.label
                    ].immutable_inputs
                ],
            }
            for scope in config.representative_scopes
        ],
        "compression_source": {
            "manifest_path": str(compression_source.manifest_path),
            "artifact_content_sha256": compression_identity[
                "content_sha256"
            ],
            "path_neutral_scientific_content_sha256": (
                compression_identity[
                    "path_neutral_scientific_content_sha256"
                ]
            ),
        },
        "resource_resume_compatibility": (
            _resource_resume_compatibility(resources)
        ),
        "candidate_device_assignments": [
            {
                "candidate_name": candidate.name,
                "devices": list(
                    _candidate_devices(
                        candidate=candidate,
                        inventory=resources,
                        safety=config.resource_performance_safety,
                    )
                ),
            }
            for candidate in config.candidates
        ],
        "execution_schedule": copy.deepcopy(dict(execution_schedule)),
        "producer_code_evidence": _benchmark_resume_code_evidence(),
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _validate_benchmark_request(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "config",
        "config_sha256",
        "workload_binding",
        "immutable_inputs_by_scope",
        "compression_source",
        "resource_resume_compatibility",
        "candidate_device_assignments",
        "execution_schedule",
        "producer_code_evidence",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("benchmark resume request does not match its closed schema")
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        value.get("schema_version") != ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA
        or value.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("benchmark resume request identity is invalid")
    if dict(value) != dict(expected):
        raise ValueError(
            "benchmark resume requires the identical immutable request"
        )


def _paused_benchmark_result(
    *,
    request_sha256: str,
    execution_schedule: Mapping[str, Any],
    completed_observation_count: int,
) -> dict[str, Any]:
    total = len(execution_schedule["entries"])
    completed = int(completed_observation_count)
    if completed < 1 or completed > total:
        raise ValueError("paused benchmark observation coverage is invalid")
    body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_PAUSED_RESULT_SCHEMA,
        "status": "paused",
        "request_sha256": request_sha256,
        "execution_schedule_content_sha256": execution_schedule[
            "content_sha256"
        ],
        "completed_observation_count": completed,
        "total_observation_count": total,
        "last_completed_sequence_index": completed - 1,
        "next_sequence_index": (
            None if completed == total else completed
        ),
        "terminal_benchmark_result_published": False,
        "resume_requires_identical_immutable_request": True,
        "operational_stop_excluded_from_request_identity": True,
    }
    return {**body, "content_sha256": identity_sha256(body)}


def _observation_root(
    destination: Path,
    entry: Mapping[str, Any],
) -> Path:
    kind = str(entry["observation_kind"])
    index = int(entry["observation_index"])
    return (
        destination
        / ("warmups" if kind == "warmup" else "runs")
        / str(entry["candidate_name"])
        / str(entry["scope_label"])
        / (
            f"warmup_{index:03d}"
            if kind == "warmup"
            else f"repetition_{index:03d}"
        )
    ).resolve()


def _observation_checkpoint_path(
    destination: Path,
    *,
    sequence_index: int,
) -> Path:
    return (
        destination
        / "checkpoints"
        / f"observation_{int(sequence_index):06d}.json"
    )


def _completed_target_from_instance(
    *,
    instance: _InstanceResult,
    workload: RoleNeutralBenchmarkWorkload,
) -> _CompletedArtifactTarget:
    if instance.manifest is None:
        raise ValueError("cannot checkpoint an incomplete benchmark fit")
    manifest_identity = instance.manifest.get("content_sha256")
    scientific_identity = instance.scientific_artifact_sha256
    return _CompletedArtifactTarget(
        root=instance.root,
        workload=workload,
        manifest_content_sha256=str(manifest_identity),
        scientific_artifact_sha256=str(scientific_identity),
    )


def _write_observation_checkpoint(
    *,
    path: Path,
    request_sha256: str,
    schedule_entry: Mapping[str, Any],
    observation: BenchmarkRunObservation,
    detail: Mapping[str, Any],
    targets: Sequence[_CompletedArtifactTarget],
    destination: Path,
) -> None:
    observation_root = Path(str(observation.artifact_path))
    checkpoint_ledger = TelemetryLedger()
    tree_sha256, total_file_bytes, file_count = _audit_tree(
        observation_root,
        ledger=checkpoint_ledger,
    )
    body = {
        "schema_version": (
            ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA
        ),
        "request_sha256": request_sha256,
        "schedule_entry": copy.deepcopy(dict(schedule_entry)),
        "observation": asdict(observation),
        "detail": copy.deepcopy(dict(detail)),
        "observation_tree": {
            "tree_sha256": tree_sha256,
            "total_file_bytes": total_file_bytes,
            "file_count": file_count,
        },
        "complete_artifacts": [
            {
                "relative_root": target.root.relative_to(
                    destination
                ).as_posix(),
                "manifest_content_sha256": (
                    target.manifest_content_sha256
                ),
                "scientific_artifact_sha256": (
                    target.scientific_artifact_sha256
                ),
            }
            for target in targets
        ],
    }
    _write_result(
        path,
        {**body, "content_sha256": identity_sha256(body)},
    )


def _validate_checkpoint_lane_interval_telemetry(
    *,
    detail: Mapping[str, Any],
    schedule_entry: Mapping[str, Any],
    candidate: RoleNeutralBenchmarkCandidate,
    configured_fits_per_observation: int,
    completed_scope_fits: int,
    component_intervals_by_fit: Sequence[
        tuple[int, Sequence[Mapping[str, Any]]]
    ],
) -> None:
    """Rebind sealed lane telemetry to freshly reopened fit artifacts."""

    accelerator_required = candidate.accelerator_count > 0
    complete_accelerator_observation = (
        accelerator_required
        and completed_scope_fits == configured_fits_per_observation
    )
    observation_value = detail.get(
        "cpu_gpu_lane_interval_observation"
    )
    analysis_value = detail.get("cpu_gpu_lane_overlap_analysis")
    if (
        detail.get("cpu_gpu_lane_interval_telemetry_required")
        is not accelerator_required
        or detail.get("cpu_gpu_lane_interval_semantics")
        != (
            "direct_monotonic_architecture_phase_envelopes_"
            "not_kernel_occupancy_v1"
        )
        or detail.get("cpu_gpu_lane_overlap_speedup_claimed") is not False
        or len(component_intervals_by_fit) != completed_scope_fits
    ):
        raise ValueError(
            "benchmark checkpoint lane-interval telemetry changed"
        )
    if not accelerator_required:
        if (
            detail.get("cpu_gpu_lane_interval_telemetry_complete") is not True
            or observation_value is not None
            or analysis_value is not None
            or detail.get("cpu_gpu_lane_overlap_descriptive_only") is not False
        ):
            raise ValueError(
                "CPU-only benchmark checkpoint invented GPU-lane analysis"
            )
        return
    if not complete_accelerator_observation:
        if (
            detail.get("cpu_gpu_lane_interval_telemetry_complete") is not False
            or observation_value is not None
            or analysis_value is not None
            or detail.get("cpu_gpu_lane_overlap_descriptive_only") is not False
        ):
            raise ValueError(
                "incomplete accelerator checkpoint claimed complete lane telemetry"
            )
        return
    expected_fit_indices = tuple(
        range(configured_fits_per_observation)
    )
    if tuple(index for index, _rows in component_intervals_by_fit) != (
        expected_fit_indices
    ):
        raise ValueError(
            "accelerator checkpoint lane intervals omit or reorder fits"
        )
    owner_execution_ids = tuple(
        f"fit_{index:03d}" for index in expected_fit_indices
    )
    expected_intervals = tuple(
        interval
        for fit_index, component_rows in component_intervals_by_fit
        for interval in _fit_lane_intervals_from_component_execution(
            owner_execution_id=f"fit_{fit_index:03d}",
            component_execution_intervals=component_rows,
        )
    )
    if (
        detail.get("cpu_gpu_lane_interval_telemetry_complete") is not True
        or detail.get("cpu_gpu_lane_overlap_descriptive_only") is not True
        or not isinstance(observation_value, Mapping)
        or not isinstance(analysis_value, Mapping)
    ):
        raise ValueError(
            "complete accelerator checkpoint omitted lane telemetry"
        )
    closed_observation = CompletedFitIntervalObservation.from_mapping(
        observation_value
    )
    expected_observation_id = (
        f"{schedule_entry['observation_kind']}:"
        f"{candidate.name}:{schedule_entry['scope_label']}:"
        f"{schedule_entry['observation_index']}"
    )
    if (
        closed_observation.observation_id != expected_observation_id
        or closed_observation.owner_execution_ids != owner_execution_ids
        or closed_observation.clock_domain_id
        != ROLE_NEUTRAL_COMPONENT_EXECUTION_CLOCK_DOMAIN
        or closed_observation.intervals != expected_intervals
    ):
        raise ValueError(
            "checkpoint lane observation differs from reopened fit intervals"
        )
    recomputed_analysis = analyze_completed_fit_lane_overlap(
        closed_observation,
        expected_observation_id=expected_observation_id,
        expected_owner_execution_ids=owner_execution_ids,
    )
    if recomputed_analysis.as_dict() != dict(analysis_value):
        raise ValueError(
            "checkpoint lane-overlap analysis differs from direct intervals"
        )


def _load_observation_checkpoint(
    *,
    path: Path,
    request_sha256: str,
    schedule_entry: Mapping[str, Any],
    destination: Path,
    workload: RoleNeutralBenchmarkWorkload,
    candidate: RoleNeutralBenchmarkCandidate,
    inventory: ResourceInventory,
    resource_performance_safety: ResourcePerformanceSafetyPolicy,
    configured_fits_per_observation: int,
) -> tuple[
    BenchmarkRunObservation,
    dict[str, Any],
    tuple[_CompletedArtifactTarget, ...],
]:
    value = _read_closed_json(path, label="benchmark observation checkpoint")
    required = {
        "schema_version",
        "request_sha256",
        "schedule_entry",
        "observation",
        "detail",
        "observation_tree",
        "complete_artifacts",
        "content_sha256",
    }
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        set(value) != required
        or value.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA
        or value.get("request_sha256") != request_sha256
        or value.get("schedule_entry") != dict(schedule_entry)
        or value.get("content_sha256") != identity_sha256(body)
        or candidate.name != schedule_entry.get("candidate_name")
    ):
        raise ValueError("benchmark observation checkpoint is invalid or unrelated")
    raw_observation = value.get("observation")
    expected_observation_fields = {
        field.name for field in fields(BenchmarkRunObservation)
    }
    if (
        not isinstance(raw_observation, Mapping)
        or set(raw_observation) != expected_observation_fields
    ):
        raise ValueError("benchmark checkpoint observation is not closed")
    observation = BenchmarkRunObservation(**dict(raw_observation))
    expected_root = _observation_root(destination, schedule_entry)
    if (
        observation.candidate_name != schedule_entry["candidate_name"]
        or observation.scope_label != schedule_entry["scope_label"]
        or observation.repetition_index
        != schedule_entry["observation_index"]
        or observation.artifact_path != str(expected_root)
        or observation.completed_scope_fits
        > int(configured_fits_per_observation)
    ):
        raise ValueError("benchmark checkpoint observation differs from its schedule")
    detail = value.get("detail")
    if (
        not isinstance(detail, dict)
        or detail.get("candidate_name") != schedule_entry["candidate_name"]
        or detail.get("scope_label") != schedule_entry["scope_label"]
        or detail.get("observation_kind")
        != schedule_entry["observation_kind"]
        or detail.get("repetition_index")
        != schedule_entry["observation_index"]
        or detail.get("execution_sequence_index")
        != schedule_entry["sequence_index"]
        or detail.get("candidate_position_within_rotation")
        != schedule_entry["candidate_position"]
        or detail.get("candidate_rotation_offset")
        != schedule_entry["rotation_offset"]
        or detail.get("scientific_artifact_sha256")
        != observation.scientific_artifact_sha256
        or detail.get("complete_scientific_artifacts_exactly_equal")
        is not observation.complete_artifacts_exactly_equal
    ):
        raise ValueError("benchmark checkpoint telemetry differs from its schedule")
    raw_tree = value.get("observation_tree")
    if not isinstance(raw_tree, Mapping) or set(raw_tree) != {
        "tree_sha256",
        "total_file_bytes",
        "file_count",
    }:
        raise ValueError("benchmark checkpoint observation-tree proof is invalid")
    if (
        expected_root.is_symlink()
        or expected_root.resolve(strict=True) != expected_root
        or not expected_root.is_dir()
    ):
        raise ValueError("benchmark checkpoint observation root is unsafe")
    checkpoint_ledger = TelemetryLedger()
    tree_sha256, total_file_bytes, file_count = _audit_tree(
        expected_root,
        ledger=checkpoint_ledger,
    )
    if dict(raw_tree) != {
        "tree_sha256": tree_sha256,
        "total_file_bytes": total_file_bytes,
        "file_count": file_count,
    }:
        raise ValueError("benchmark checkpoint observation tree changed")
    raw_targets = value.get("complete_artifacts")
    if (
        not isinstance(raw_targets, list)
        or len(raw_targets) != observation.completed_scope_fits
    ):
        raise ValueError("benchmark checkpoint artifact coverage is incomplete")
    targets: list[_CompletedArtifactTarget] = []
    expected_roots = tuple(
        (expected_root / f"fit_{index:03d}").resolve()
        for index in range(int(configured_fits_per_observation))
    )
    candidate_devices = _candidate_devices(
        candidate=candidate,
        inventory=inventory,
        safety=resource_performance_safety,
    )
    primary_slots = tuple(
        device
        for device in candidate_devices
        for _index in range(candidate.concurrency_per_device)
    )
    runtime_topologies = candidate.neural_query_topology.runtime_topologies(
        candidate_devices
    )
    component_intervals_by_fit: list[
        tuple[int, tuple[Mapping[str, Any], ...]]
    ] = []
    previous_fit_index = -1
    for row in raw_targets:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_root",
            "manifest_content_sha256",
            "scientific_artifact_sha256",
        }:
            raise ValueError("benchmark checkpoint artifact row is invalid")
        relative = PurePosixPath(str(row["relative_root"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("benchmark checkpoint artifact path is unsafe")
        root = (destination / Path(*relative.parts)).resolve(strict=True)
        if root not in expected_roots:
            raise ValueError(
                "benchmark checkpoint artifact points outside configured fits"
            )
        fit_index = expected_roots.index(root)
        if fit_index <= previous_fit_index:
            raise ValueError(
                "benchmark checkpoint artifact rows changed canonical fit order"
            )
        previous_fit_index = fit_index
        target = _CompletedArtifactTarget(
            root=root,
            workload=workload,
            manifest_content_sha256=str(
                row["manifest_content_sha256"]
            ),
            scientific_artifact_sha256=str(
                row["scientific_artifact_sha256"]
            ),
        )
        validated = validate_role_neutral_stage1_execution(
            root=root,
            plan=workload.plan,
        )
        scientific = validated.get("scientific_identity")
        if (
            validated.get("content_sha256")
            != target.manifest_content_sha256
            or not isinstance(scientific, Mapping)
            or scientific.get("content_sha256")
            != target.scientific_artifact_sha256
            or (
                observation.scientific_artifact_sha256 is not None
                and target.scientific_artifact_sha256
                != observation.scientific_artifact_sha256
            )
        ):
            raise ValueError(
                "benchmark checkpoint complete artifact changed"
            )
        primary_resource = primary_slots[fit_index % len(primary_slots)]
        component_intervals_by_fit.append(
            (
                fit_index,
                _component_execution_intervals_from_manifest(
                    manifest=validated,
                    expected_physical_owner_scope_id=(
                        workload.plan.physical_scopes[0].scope_id
                    ),
                    expected_primary_resource=primary_resource,
                    expected_neural_query_resources=(
                        runtime_topologies[primary_resource].devices
                    ),
                ),
            )
        )
        targets.append(target)
    target_roots = {target.root for target in targets}
    if (
        not target_roots.issubset(set(expected_roots))
        or (
            observation.complete_artifacts_exactly_equal
            and (
                target_roots != set(expected_roots)
                or observation.scientific_artifact_sha256 is None
            )
        )
    ):
        raise ValueError("benchmark checkpoint artifact roots are incomplete")
    _validate_checkpoint_lane_interval_telemetry(
        detail=detail,
        schedule_entry=schedule_entry,
        candidate=candidate,
        configured_fits_per_observation=(
            configured_fits_per_observation
        ),
        completed_scope_fits=observation.completed_scope_fits,
        component_intervals_by_fit=component_intervals_by_fit,
    )
    return observation, copy.deepcopy(detail), tuple(targets)


def _recover_interrupted_observation(
    *,
    destination: Path,
    schedule_entry: Mapping[str, Any],
    request_sha256: str,
) -> None:
    """Preserve and authenticate an uncheckpointed attempt before retrying."""

    source = _observation_root(destination, schedule_entry)
    if not source.exists() and not source.is_symlink():
        return
    if (
        source.is_symlink()
        or source.resolve(strict=True) != source
        or not source.is_dir()
    ):
        raise ValueError("interrupted benchmark observation root is unsafe")
    ledger = TelemetryLedger()
    tree_sha256, total_bytes, file_count = _audit_tree(
        source,
        ledger=ledger,
    )
    interrupted = destination / "interrupted_observations"
    sequence_index = int(schedule_entry["sequence_index"])
    attempt_index = 0
    while True:
        target = interrupted / (
            f"observation_{sequence_index:06d}_attempt_{attempt_index:03d}"
        )
        attestation_path = target.with_suffix(".json")
        if (
            not target.exists()
            and not target.is_symlink()
            and not attestation_path.exists()
            and not attestation_path.is_symlink()
        ):
            break
        attempt_index += 1
    os.replace(source, target)
    for directory in {source.parent, target.parent}:
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    body = {
        "schema_version": (
            ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA
        ),
        "request_sha256": request_sha256,
        "schedule_entry": copy.deepcopy(dict(schedule_entry)),
        "preserved_relative_root": target.relative_to(
            destination
        ).as_posix(),
        "tree_sha256": tree_sha256,
        "total_file_bytes": total_bytes,
        "file_count": file_count,
    }
    _write_result(
        attestation_path,
        {**body, "content_sha256": identity_sha256(body)},
    )


def _validate_interrupted_observations(
    *,
    destination: Path,
    execution_schedule: Mapping[str, Any],
    request_sha256: str,
) -> None:
    root = destination / "interrupted_observations"
    entries_by_sequence = {
        int(entry["sequence_index"]): dict(entry)
        for entry in execution_schedule["entries"]
    }
    children = tuple(root.iterdir())
    if any(
        path.is_symlink()
        or (
            not path.is_dir()
            and not (path.is_file() and path.suffix == ".json")
        )
        for path in children
    ):
        raise ValueError(
            "interrupted benchmark observation directory contains unrelated data"
        )
    directories = {
        path.name: path
        for path in children
        if path.is_dir() and not path.is_symlink()
    }
    attestations = {
        path.stem: path
        for path in children
        if path.is_file()
        and not path.is_symlink()
        and path.suffix == ".json"
    }
    if set(directories) != set(attestations):
        raise ValueError(
            "interrupted benchmark observations lack exact attestations"
        )
    pattern = re.compile(r"^observation_([0-9]{6})_attempt_([0-9]{3})$")
    for name, preserved_root in sorted(directories.items()):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(
                "interrupted benchmark observation name is invalid"
            )
        sequence_index = int(match.group(1))
        expected_entry = entries_by_sequence.get(sequence_index)
        value = _read_closed_json(
            attestations[name],
            label="interrupted benchmark observation attestation",
        )
        required = {
            "schema_version",
            "request_sha256",
            "schedule_entry",
            "preserved_relative_root",
            "tree_sha256",
            "total_file_bytes",
            "file_count",
            "content_sha256",
        }
        body = {
            key: copy.deepcopy(child)
            for key, child in value.items()
            if key != "content_sha256"
        }
        if (
            set(value) != required
            or value.get("schema_version")
            != ROLE_NEUTRAL_BENCHMARK_INTERRUPTED_OBSERVATION_SCHEMA
            or value.get("request_sha256") != request_sha256
            or expected_entry is None
            or value.get("schedule_entry") != expected_entry
            or value.get("preserved_relative_root")
            != preserved_root.relative_to(destination).as_posix()
            or value.get("content_sha256") != identity_sha256(body)
        ):
            raise ValueError(
                "interrupted benchmark observation attestation is invalid"
            )
        ledger = TelemetryLedger()
        tree_sha256, total_bytes, file_count = _audit_tree(
            preserved_root,
            ledger=ledger,
        )
        if (
            value.get("tree_sha256") != tree_sha256
            or value.get("total_file_bytes") != total_bytes
            or value.get("file_count") != file_count
        ):
            raise ValueError(
                "interrupted benchmark observation bytes changed"
            )


def _validate_observation_root_coverage(
    *,
    destination: Path,
    execution_schedule: Mapping[str, Any],
) -> None:
    expected = {
        _observation_root(destination, entry)
        for entry in execution_schedule["entries"]
    }
    for base_name in ("warmups", "runs"):
        base = destination / base_name
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError(
                    "benchmark observation trees contain a symlink"
                )
            resolved = path.resolve(strict=True)
            if not any(
                resolved == root
                or resolved in root.parents
                or root in resolved.parents
                for root in expected
            ):
                raise ValueError(
                    "benchmark observation trees contain unrelated data"
                )
    missing = [
        str(root)
        for root in sorted(expected)
        if not root.is_dir() or root.is_symlink()
    ]
    if missing:
        raise ValueError(
            "benchmark observation tree coverage is incomplete: "
            f"{missing}"
        )


def _load_compression_benchmark(
    *,
    path: Path,
    config: RoleNeutralBenchmarkConfig,
    source: PortableProductionStage1ClusterPreflightArtifact,
) -> dict[str, Any]:
    value = _read_closed_json(
        path,
        label="compact-preflight compression benchmark result",
    )
    validated = validate_compact_preflight_compression_benchmark_result(
        value,
        reopen_artifacts=True,
    )
    source_identity = source.identity()
    registered_source = validated.get("source")
    if (
        validated.get("config")
        != config.preflight_compression_benchmark.as_dict()
        or not isinstance(registered_source, Mapping)
        or registered_source.get("artifact_content_sha256")
        != source_identity["content_sha256"]
        or registered_source.get(
            "path_neutral_scientific_content_sha256"
        )
        != source_identity["path_neutral_scientific_content_sha256"]
    ):
        raise ValueError(
            "compression benchmark differs from the immutable resume request"
        )
    return validated


def _close_physical_owner_executor(executor: Any) -> None:
    """Close one benchmark-owned executor session when it exposes a close seam."""

    close = getattr(executor, "close", None)
    if close is not None:
        if not callable(close):
            raise TypeError("benchmark executor close attribute is not callable")
        close()


def run_role_neutral_performance_benchmark(
    *,
    config: RoleNeutralBenchmarkConfig,
    workloads: Mapping[str, RoleNeutralBenchmarkWorkload],
    output_root: Path | str,
    inventory: ResourceInventory | None = None,
    resume: bool = False,
    stop_after_completed_observations: int | None = None,
) -> dict[str, Any]:
    """Run or explicitly resume configured real role-neutral fits.

    Resume is accepted only when the root's sealed request is byte-logically
    identical to the newly constructed request.  Every skipped observation is
    reopened through its production validator before its checkpoint is used.
    """

    if not isinstance(config, RoleNeutralBenchmarkConfig):
        raise TypeError("benchmark runner requires a typed config")
    if type(resume) is not bool:
        raise TypeError("benchmark resume must be boolean")
    if stop_after_completed_observations is not None and (
        isinstance(stop_after_completed_observations, bool)
        or not isinstance(stop_after_completed_observations, int)
        or stop_after_completed_observations < 1
    ):
        raise ValueError(
            "stop_after_completed_observations must be a positive integer"
        )
    destination = Path(output_root)
    if not destination.is_absolute():
        raise ValueError("benchmark output_root must be absolute")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError("benchmark output parent must be canonical")
    destination_exists = destination.exists() or destination.is_symlink()
    if destination_exists and not resume:
        raise FileExistsError("benchmark output_root must be fresh")
    if not destination_exists and resume:
        raise FileNotFoundError("benchmark resume output_root does not exist")
    if destination_exists and (
        destination.is_symlink()
        or destination.resolve(strict=True) != destination
        or not destination.is_dir()
    ):
        raise ValueError("benchmark resume output_root must be canonical")
    configured_labels = {value.label for value in config.representative_scopes}
    if not isinstance(workloads, Mapping) or set(workloads) != configured_labels:
        raise ValueError("benchmark workloads do not match configured scopes")
    typed_workloads: dict[str, RoleNeutralBenchmarkWorkload] = {}
    for scope in config.representative_scopes:
        workload = workloads[scope.label]
        if (
            not isinstance(workload, RoleNeutralBenchmarkWorkload)
            or workload.scope_label != scope.label
            or workload.fit_row_count != scope.fit_row_count
        ):
            raise ValueError(
                f"benchmark workload {scope.label!r} differs from configured size"
            )
        typed_workloads[scope.label] = workload
    scientific_htr_batches = {
        value.scientific_htr_training_batch_size
        for value in typed_workloads.values()
    }
    configured_htr_batches = {
        value.htr_operational_controls.training_batch_size
        for value in config.candidates
    }
    if (
        len(scientific_htr_batches) != 1
        or configured_htr_batches != scientific_htr_batches
    ):
        raise ValueError(
            "benchmark HTR training_batch_size binding differs from the "
            "authenticated prepared scientific profile; optimizer batches "
            "are not deployment-tunable"
        )
    source_bindings = {
        identity_sha256(value.source_binding.as_dict()): value.source_binding
        for value in typed_workloads.values()
    }
    if len(source_bindings) != 1:
        raise ValueError(
            "benchmark representative workloads name different staged sources"
        )
    [source_binding] = source_bindings.values()
    workload_binding_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
        "source": source_binding.as_dict(),
        "representative_scope_plans": [
            {
                "scope_label": scope.label,
                "fit_row_count": scope.fit_row_count,
                "plan_scientific_content_sha256": (
                    typed_workloads[scope.label].plan.scientific_content_sha256
                ),
                "physical_owner_scope_id": (
                    typed_workloads[scope.label].plan.physical_scopes[0].scope_id
                ),
            }
            for scope in config.representative_scopes
        ],
    }
    workload_binding = {
        **workload_binding_body,
        "content_sha256": identity_sha256(workload_binding_body),
    }
    resources = inventory or discover_resources()
    if not isinstance(resources, ResourceInventory):
        raise TypeError("benchmark resource discovery returned an untyped inventory")
    if max(value.host_cpu_budget for value in config.candidates) > resources.cpu_count:
        raise RuntimeError("benchmark candidate CPU budget exceeds the host inventory")
    compression_sources = [
        typed_workloads[label].preflight_compression_source_builder()
        for label in sorted(typed_workloads)
    ]
    if any(
        not isinstance(
            value,
            PortableProductionStage1ClusterPreflightArtifact,
        )
        for value in compression_sources
    ):
        raise TypeError(
            "benchmark workload returned an untyped compact-preflight source"
        )
    compression_source_keys = {
        (
            value.identity()["content_sha256"],
            str(value.manifest_path),
        )
        for value in compression_sources
    }
    if len(compression_source_keys) != 1:
        raise ValueError(
            "representative workloads do not share one exact compact preflight"
        )
    execution_schedule = _benchmark_execution_schedule(config)
    if (
        stop_after_completed_observations is not None
        and stop_after_completed_observations
        > len(execution_schedule["entries"])
    ):
        raise ValueError(
            "stop_after_completed_observations exceeds the configured schedule"
        )
    benchmark_request = _benchmark_request(
        config=config,
        workload_binding=workload_binding,
        typed_workloads=typed_workloads,
        compression_source=compression_sources[0],
        resources=resources,
        execution_schedule=execution_schedule,
    )
    request_path = destination / "benchmark_request.json"
    if not resume:
        destination.mkdir(exist_ok=False)
        for child in (
            "warmups",
            "runs",
            "executor_sessions",
            "checkpoints",
            "interrupted_observations",
        ):
            (destination / child).mkdir(exist_ok=False)
        _write_result(request_path, benchmark_request)
    else:
        supplied_request = _read_closed_json(
            request_path,
            label="benchmark resume request",
        )
        _validate_benchmark_request(
            supplied_request,
            expected=benchmark_request,
        )
        for child in (
            "warmups",
            "runs",
            "executor_sessions",
            "checkpoints",
            "interrupted_observations",
        ):
            path = destination / child
            if (
                path.is_symlink()
                or path.resolve(strict=True) != path
                or not path.is_dir()
            ):
                raise ValueError(
                    "benchmark resume root has a missing or unsafe directory"
                )
        if (destination / "benchmark_result.json").exists():
            raise RuntimeError("benchmark result is already complete")
    compression_root = (
        destination / "preflight_compression_benchmark"
    ).resolve()
    compression_result_path = (
        compression_root / "compression_benchmark_result.json"
    )
    if resume and compression_root.exists():
        if not compression_result_path.exists():
            raise RuntimeError(
                "incomplete compact-preflight compression benchmark "
                "cannot be adopted"
            )
        compression_benchmark = _load_compression_benchmark(
            path=compression_result_path,
            config=config,
            source=compression_sources[0],
        )
    else:
        compression_benchmark = (
            run_compact_preflight_compression_benchmark(
                config=config.preflight_compression_benchmark,
                source=compression_sources[0],
                output_root=compression_root,
            )
        )
    expected_top_level = {
        "benchmark_request.json",
        "warmups",
        "runs",
        "executor_sessions",
        "checkpoints",
        "interrupted_observations",
        "preflight_compression_benchmark",
    }
    if {path.name for path in destination.iterdir()} != expected_top_level:
        raise ValueError("benchmark output root contains unrelated data")
    if any((destination / "executor_sessions").iterdir()):
        raise RuntimeError(
            "benchmark output retains an unclosed executor session"
        )
    _validate_interrupted_observations(
        destination=destination,
        execution_schedule=execution_schedule,
        request_sha256=str(benchmark_request["content_sha256"]),
    )

    warmup_observations: list[BenchmarkRunObservation] = []
    warmup_details: list[dict[str, Any]] = []
    observations: list[BenchmarkRunObservation] = []
    details: list[dict[str, Any]] = []
    audit_targets: list[_CompletedArtifactTarget] = []
    candidates = tuple(config.candidates)
    candidate_by_name = {value.name: value for value in candidates}
    scope_by_label = {
        value.label: value for value in config.representative_scopes
    }
    completed_observation_count = 0
    checkpoint_directory = destination / "checkpoints"
    expected_checkpoint_names = {
        _observation_checkpoint_path(
            destination,
            sequence_index=int(entry["sequence_index"]),
        ).name
        for entry in execution_schedule["entries"]
    }
    existing_checkpoint_indices: list[int] = []
    checkpoint_pattern = re.compile(r"^observation_([0-9]{6})\.json$")
    for path in checkpoint_directory.iterdir():
        match = checkpoint_pattern.fullmatch(path.name)
        if (
            path.name not in expected_checkpoint_names
            or match is None
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(
                "benchmark checkpoint directory contains unrelated data"
            )
        existing_checkpoint_indices.append(int(match.group(1)))
    existing_checkpoint_indices.sort()
    if existing_checkpoint_indices != list(
        range(len(existing_checkpoint_indices))
    ):
        raise ValueError(
            "benchmark observation checkpoints are not one ordered prefix"
        )
    if (
        stop_after_completed_observations is not None
        and len(existing_checkpoint_indices)
        > stop_after_completed_observations
    ):
        raise ValueError(
            "observation stop precedes already sealed checkpoint coverage"
        )

    for entry in execution_schedule["entries"]:
        sequence_index = int(entry["sequence_index"])
        scope = scope_by_label[str(entry["scope_label"])]
        candidate = candidate_by_name[str(entry["candidate_name"])]
        workload = typed_workloads[scope.label]
        checkpoint_path = _observation_checkpoint_path(
            destination,
            sequence_index=sequence_index,
        )
        if checkpoint_path.exists():
            observation, enriched_detail, targets = (
                _load_observation_checkpoint(
                    path=checkpoint_path,
                    request_sha256=str(
                        benchmark_request["content_sha256"]
                    ),
                    schedule_entry=entry,
                    destination=destination,
                    workload=workload,
                    candidate=candidate,
                    inventory=resources,
                    resource_performance_safety=(
                        config.resource_performance_safety
                    ),
                    configured_fits_per_observation=(
                        scope.fits_per_observation
                    ),
                )
            )
        else:
            if resume:
                _recover_interrupted_observation(
                    destination=destination,
                    schedule_entry=entry,
                    request_sha256=str(
                        benchmark_request["content_sha256"]
                    ),
                )
            base_executor = workload.physical_owner_executor_builder(
                candidate.executor_mode,
                candidate.concurrency_per_device,
            )
            executor = base_executor
            session_marker_root: Path | None = None
            if candidate.executor_mode == "persistent_slots":
                open_session = getattr(base_executor, "open_session", None)
                if not callable(open_session):
                    raise TypeError(
                        "persistent benchmark candidate requires an executor "
                        "with open_session()"
                    )
                session_marker_root = (
                    destination
                    / "executor_sessions"
                    / f"observation_{sequence_index:06d}"
                ).resolve()
                executor = open_session(
                    resources=_candidate_devices(
                        candidate=candidate,
                        inventory=resources,
                        safety=config.resource_performance_safety,
                    ),
                    max_workers=candidate.total_concurrency,
                    cpu_budget=candidate.host_cpu_budget,
                    marker_root=session_marker_root,
                )
            execution_failure: BaseException | None = None
            observation_result: tuple[
                BenchmarkRunObservation,
                dict[str, Any],
                tuple[_InstanceResult, ...],
            ] | None = None
            try:
                observation_result = _run_observation(
                    root=destination,
                    config=config,
                    candidate=candidate,
                    scope=scope,
                    workload=workload,
                    repetition_index=int(entry["observation_index"]),
                    inventory=resources,
                    physical_owner_executor=executor,
                    observation_kind=str(entry["observation_kind"]),
                )
            except BaseException as exc:
                execution_failure = exc
            finally:
                try:
                    _close_physical_owner_executor(executor)
                except BaseException as exc:
                    if execution_failure is None:
                        execution_failure = RuntimeError(
                            "benchmark could not close its observation-owned "
                            "executor session"
                        )
                        execution_failure.__cause__ = exc
            if (
                session_marker_root is not None
                and (
                    session_marker_root.exists()
                    or session_marker_root.is_symlink()
                )
                and execution_failure is None
            ):
                execution_failure = RuntimeError(
                    "benchmark executor close left its session marker root"
                )
            if execution_failure is not None:
                raise execution_failure
            if observation_result is None:
                raise RuntimeError(
                    "benchmark observation completed without a result"
                )
            observation, detail, instances = observation_result
            enriched_detail = {
                **detail,
                "execution_sequence_index": sequence_index,
                "candidate_position_within_rotation": int(
                    entry["candidate_position"]
                ),
                "candidate_rotation_offset": int(
                    entry["rotation_offset"]
                ),
            }
            targets = tuple(
                _completed_target_from_instance(
                    instance=instance,
                    workload=workload,
                )
                for instance in instances
                if instance.manifest is not None
            )
            _write_observation_checkpoint(
                path=checkpoint_path,
                request_sha256=str(
                    benchmark_request["content_sha256"]
                ),
                schedule_entry=entry,
                observation=observation,
                detail=enriched_detail,
                targets=targets,
                destination=destination,
            )
        if entry["observation_kind"] == "warmup":
            warmup_observations.append(observation)
            warmup_details.append(enriched_detail)
        else:
            observations.append(observation)
            details.append(enriched_detail)
        audit_targets.extend(targets)
        completed_observation_count += 1
        if (
            stop_after_completed_observations is not None
            and completed_observation_count
            == stop_after_completed_observations
        ):
            sealed_observation, sealed_detail, sealed_targets = (
                _load_observation_checkpoint(
                    path=checkpoint_path,
                    request_sha256=str(
                        benchmark_request["content_sha256"]
                    ),
                    schedule_entry=entry,
                    destination=destination,
                    workload=workload,
                    candidate=candidate,
                    inventory=resources,
                    resource_performance_safety=(
                        config.resource_performance_safety
                    ),
                    configured_fits_per_observation=(
                        scope.fits_per_observation
                    ),
                )
            )
            if (
                sealed_observation != observation
                or sealed_detail != enriched_detail
                or sealed_targets != targets
            ):
                raise RuntimeError(
                    "paused benchmark checkpoint changed after sealing"
                )
            if any((destination / "executor_sessions").iterdir()):
                raise RuntimeError(
                    "paused benchmark retains an executor session"
                )
            _validate_interrupted_observations(
                destination=destination,
                execution_schedule=execution_schedule,
                request_sha256=str(
                    benchmark_request["content_sha256"]
                ),
            )
            return _paused_benchmark_result(
                request_sha256=str(
                    benchmark_request["content_sha256"]
                ),
                execution_schedule=execution_schedule,
                completed_observation_count=(
                    completed_observation_count
                ),
            )

    _validate_observation_root_coverage(
        destination=destination,
        execution_schedule=execution_schedule,
    )
    _validate_interrupted_observations(
        destination=destination,
        execution_schedule=execution_schedule,
        request_sha256=str(benchmark_request["content_sha256"]),
    )
    terminal_audit, terminal_telemetry = _terminal_audit(
        completed=audit_targets,
    )
    candidate_rows, _raw_selected, scientific_identity = (
        _candidate_benchmark_summaries(
            policy=config.acceptance_policy,
            benchmark_runs=observations,
        )
    )
    detail_by_candidate_scope_repetition = {
        (
            str(row["candidate_name"]),
            str(row["scope_label"]),
            int(row["repetition_index"]),
        ): row
        for row in details
    }
    warmup_detail_by_candidate = {
        candidate.name: [
            row
            for row in warmup_details
            if row["candidate_name"] == candidate.name
        ]
        for candidate in candidates
    }
    measured_hashes_by_candidate_scope = {
        (candidate.name, scope.label): {
            str(row.scientific_artifact_sha256)
            for row in observations
            if row.candidate_name == candidate.name
            and row.scope_label == scope.label
            and row.scientific_artifact_sha256 is not None
        }
        for candidate in candidates
        for scope in config.representative_scopes
    }
    accepted_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        candidate_name = str(row["candidate_name"])
        configured_candidate = next(
            value for value in candidates if value.name == candidate_name
        )
        telemetry_rows = [
            value
            for key, value in detail_by_candidate_scope_repetition.items()
            if key[0] == candidate_name
        ]
        telemetry_accepted = bool(telemetry_rows) and all(
            value["telemetry_accepted"] for value in telemetry_rows
        )
        lane_interval_telemetry_accepted = bool(telemetry_rows) and all(
            (
                value.get(
                    "cpu_gpu_lane_interval_telemetry_required"
                )
                is (configured_candidate.accelerator_count > 0)
            )
            and value.get(
                "cpu_gpu_lane_interval_telemetry_complete"
            )
            is True
            and value.get(
                "cpu_gpu_lane_overlap_speedup_claimed"
            )
            is False
            and (
                (
                    isinstance(
                        value.get(
                            "cpu_gpu_lane_interval_observation"
                        ),
                        Mapping,
                    )
                    and isinstance(
                        value.get(
                            "cpu_gpu_lane_overlap_analysis"
                        ),
                        Mapping,
                    )
                    and value.get(
                        "cpu_gpu_lane_overlap_descriptive_only"
                    )
                    is True
                )
                if configured_candidate.accelerator_count > 0
                else (
                    value.get(
                        "cpu_gpu_lane_interval_observation"
                    )
                    is None
                    and value.get(
                        "cpu_gpu_lane_overlap_analysis"
                    )
                    is None
                )
            )
            for value in telemetry_rows
        )
        lane_overlap_observation_count = sum(
            isinstance(
                value.get("cpu_gpu_lane_overlap_analysis"),
                Mapping,
            )
            for value in telemetry_rows
        )
        htr_operational_attestations_accepted = bool(telemetry_rows) and all(
            len(value.get("htr_operational_attestations") or ())
            == int(value["configured_fits_per_observation"])
            and all(
                isinstance(item.get("attestation"), Mapping)
                and item["attestation"].get("controls")
                == configured_candidate.htr_operational_controls.as_dict()
                for item in value["htr_operational_attestations"]
            )
            for value in telemetry_rows
        )

        def topology_attestations_accepted(
            rows: Sequence[Mapping[str, Any]],
        ) -> bool:
            if not rows:
                return False
            for telemetry_row in rows:
                devices = tuple(
                    str(value)
                    for value in telemetry_row.get("device_ids") or ()
                )
                if not devices:
                    return False
                expected_by_primary = (
                    configured_candidate.neural_query_topology
                    .runtime_topologies(devices)
                )
                slots = tuple(
                    device
                    for device in devices
                    for _index in range(
                        configured_candidate.concurrency_per_device
                    )
                )
                attestations = telemetry_row.get(
                    "neural_query_topology_attestations"
                )
                if (
                    not isinstance(attestations, list)
                    or len(attestations)
                    != int(
                        telemetry_row["configured_fits_per_observation"]
                    )
                ):
                    return False
                for item in attestations:
                    if not isinstance(item, Mapping):
                        return False
                    fit_index = item.get("fit_index")
                    attestation = item.get("attestation")
                    if (
                        isinstance(fit_index, bool)
                        or not isinstance(fit_index, int)
                        or fit_index < 0
                        or not isinstance(attestation, Mapping)
                    ):
                        return False
                    primary = slots[fit_index % len(slots)]
                    expected = expected_by_primary[primary]
                    if (
                        attestation.get("devices")
                        != list(expected.devices)
                        or attestation.get("homogeneous") is not True
                        or attestation.get(
                            "scientific_identity_includes_topology"
                        )
                        is not False
                    ):
                        return False
            return True

        measured_topology_attested = topology_attestations_accepted(
            telemetry_rows
        )
        candidate_warmup_rows = warmup_detail_by_candidate[candidate_name]
        expected_warmup_count = (
            len(config.representative_scopes)
            * config.warmup_observations_per_candidate_scope
        )
        warmup_htr_operational_attestations_accepted = (
            len(candidate_warmup_rows) == expected_warmup_count
        ) and all(
            len(value.get("htr_operational_attestations") or ())
            == int(value["configured_fits_per_observation"])
            and all(
                isinstance(item.get("attestation"), Mapping)
                and item["attestation"].get("controls")
                == configured_candidate.htr_operational_controls.as_dict()
                for item in value["htr_operational_attestations"]
            )
            for value in candidate_warmup_rows
        )
        warmup_telemetry_accepted = (
            len(candidate_warmup_rows) == expected_warmup_count
            and all(value["telemetry_accepted"] for value in candidate_warmup_rows)
        )
        warmup_topology_attested = (
            True
            if expected_warmup_count == 0
            else topology_attestations_accepted(
                candidate_warmup_rows
            )
        )
        warmup_matches_measured = (
            warmup_telemetry_accepted
            and (
                expected_warmup_count == 0
                or all(
                    {
                        str(value["scientific_artifact_sha256"])
                        for value in candidate_warmup_rows
                        if value["scope_label"] == scope.label
                        and value["scientific_artifact_sha256"] is not None
                    }
                    == measured_hashes_by_candidate_scope[
                        (candidate_name, scope.label)
                    ]
                    for scope in config.representative_scopes
                )
            )
        )
        updated = {
            **row,
            "executor_mode": configured_candidate.executor_mode,
            "htr_operational_controls": (
                configured_candidate.htr_operational_controls.as_dict()
            ),
            "neural_query_topology": (
                configured_candidate.neural_query_topology.as_dict()
            ),
            "resource_slot_count": (
                configured_candidate.resource_slot_count
            ),
            "effective_parallel_owners": (
                configured_candidate.total_concurrency
            ),
            "neural_query_topology_runtime_attestations_accepted": (
                measured_topology_attested
                and warmup_topology_attested
            ),
            "htr_operational_attestations_accepted": (
                htr_operational_attestations_accepted
                and warmup_htr_operational_attestations_accepted
            ),
            "measured_observation_telemetry_accepted": telemetry_accepted,
            "cpu_gpu_lane_interval_telemetry_accepted": (
                lane_interval_telemetry_accepted
            ),
            "cpu_gpu_lane_overlap_observation_count": (
                lane_overlap_observation_count
            ),
            "cpu_gpu_lane_overlap_descriptive_only": (
                configured_candidate.accelerator_count > 0
                and lane_overlap_observation_count
                == len(telemetry_rows)
            ),
            "cpu_gpu_lane_overlap_speedup_claimed": False,
            "warmup_observation_telemetry_accepted": (
                warmup_telemetry_accepted
            ),
            "warmup_scientific_identity_matches_measured": (
                warmup_matches_measured
            ),
            "accepted": (
                bool(row["accepted"])
                and telemetry_accepted
                and lane_interval_telemetry_accepted
                and htr_operational_attestations_accepted
                and warmup_htr_operational_attestations_accepted
                and warmup_telemetry_accepted
                and warmup_matches_measured
            ),
        }
        accepted_rows.append(updated)
    selectable = [value for value in accepted_rows if value["accepted"]]
    selected = (
        None
        if (
            not selectable
            or compression_benchmark["accepted"] is not True
        )
        else min(
            selectable,
            key=lambda value: (
                -float(value["throughput_fit_rows_per_second"]),
                int(value["effective_parallel_owners"]),
                str(value["candidate_name"]),
            ),
        )["candidate_name"]
    )
    benchmark_matrix_coverage = (
        build_role_neutral_benchmark_matrix_coverage(
            config=config,
            candidate_rows=accepted_rows,
            compression_benchmark_result=compression_benchmark,
        )
    )
    result_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
        "status": "complete",
        "config": config.as_dict(),
        "config_sha256": identity_sha256(config.as_dict()),
        "workload_binding": workload_binding,
        "resource_inventory": resources.as_dict(),
        "execution_schedule": execution_schedule,
        "warmup_observations": [
            asdict(value) for value in warmup_observations
        ],
        "warmup_telemetry": warmup_details,
        "warmup_observations_excluded_from_selection": True,
        "benchmark_observations": [asdict(value) for value in observations],
        "observation_telemetry": details,
        "terminal_audit": terminal_audit,
        "terminal_audit_telemetry": terminal_telemetry,
        "ordinary_observations_exclude_terminal_audit": True,
        "candidate_results": accepted_rows,
        "preflight_compression_benchmark": compression_benchmark,
        "benchmark_matrix_coverage": benchmark_matrix_coverage,
        "selected_candidate": selected,
        "selection_policy": (
            "fastest_end_to_end_then_lower_effective_owner_concurrency_then_name_v2"
        ),
        "scientific_result_identity_sha256": scientific_identity,
        "accepted": (
            selected is not None
            and compression_benchmark["accepted"] is True
            and terminal_audit["exactly_one_completed_terminal_audit"] is True
        ),
    }
    result = {
        **result_body,
        "content_sha256": identity_sha256(result_body),
    }
    _write_result(destination / "benchmark_result.json", result)
    return copy.deepcopy(result)


__all__ = [
    "ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_OBSERVATION_CHECKPOINT_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_PAUSED_RESULT_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_REQUEST_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_MATRIX_COVERAGE_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA",
    "ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA",
    "RoleNeutralBenchmarkCandidate",
    "RoleNeutralBenchmarkConfig",
    "RoleNeutralBenchmarkScope",
    "RoleNeutralBenchmarkSourceBinding",
    "RoleNeutralBenchmarkWorkload",
    "build_role_neutral_benchmark_matrix_coverage",
    "run_role_neutral_performance_benchmark",
]
