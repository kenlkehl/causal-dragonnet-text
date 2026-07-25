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
from .production_stage1_role_neutral_execution import (
    BoundRoleNeutralComponentProducer,
    RoleNeutralProducerFactories,
    RoleNeutralStage1ExecutionPolicy,
    execute_and_publish_role_neutral_stage1,
    validate_role_neutral_stage1_execution,
)
from .production_stage1_scope_scheduler import Stage1ScopePlan
from .production_stage1_cluster_preflight_artifact_v2 import (
    PortableProductionStage1ClusterPreflightArtifact,
)
from .role_neutral_all_ten_binding import EXPECTED_COMPONENT_FAMILIES

ROLE_NEUTRAL_BENCHMARK_CONFIG_SCHEMA = (
    "portable_role_neutral_performance_benchmark_config_v3"
)
ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA = (
    "portable_role_neutral_performance_benchmark_result_v4"
)
ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA = (
    "portable_role_neutral_benchmark_execution_schedule_v2"
)
ROLE_NEUTRAL_BENCHMARK_MATRIX_COVERAGE_SCHEMA = (
    "portable_role_neutral_benchmark_matrix_coverage_v2"
)
ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA = (
    "portable_role_neutral_benchmark_source_binding_v2"
)
ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA = (
    "portable_role_neutral_benchmark_workload_binding_v1"
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
        if self.total_concurrency > self.host_cpu_budget:
            raise ValueError(
                f"candidate {self.name!r} concurrency exceeds its host CPU budget"
            )

    @property
    def total_concurrency(self) -> int:
        device_count = self.accelerator_count or 1
        return int(device_count) * int(self.concurrency_per_device)

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
        return cls(**dict(value))


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
            _positive_integer(
                self.warmup_observations_per_candidate_scope,
                label="warmup_observations_per_candidate_scope",
            ),
        )
        object.__setattr__(self, "representative_scopes", scopes)
        object.__setattr__(self, "candidates", candidates)
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
                for candidate, baseline in baselines
            )
        ):
            raise ValueError("benchmark multi-device baseline bindings are invalid")
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
            )
            for value in candidates
            if value.executor_mode == "fresh_per_fit"
        }
        persistent_shapes = {
            (
                value.accelerator_count,
                value.concurrency_per_device,
                value.host_cpu_budget,
            )
            for value in candidates
            if value.executor_mode == "persistent_slots"
        }
        if not fresh_shapes.intersection(persistent_shapes):
            raise ValueError(
                "fresh and persistent benchmark modes require at least one "
                "matched resource/concurrency/CPU candidate pair"
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
    "production_stage1_cluster_preflight_artifact_v2": (
        "production_stage1_cluster_preflight_artifact_v2.py"
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
                "scientific_configuration_not_operationally_tunable"
            ),
            "performance_claimed": False,
            "configured_candidate_values": [],
            "equality_gate": (
                "not_applicable_without_deployment_only_producer_seam"
            ),
            "reason_code": (
                "htr_batches_are_authenticated_scientific_fields_and_"
                "role_neutral_htr_has_no_data_loader_worker_seam"
            ),
            "component_dispositions": [
                {
                    "component": "htr_training_batch_size",
                    "disposition": (
                        "scientific_configuration_not_operationally_tunable"
                    ),
                },
                {
                    "component": "htr_sentence_encoder_batch_size",
                    "disposition": (
                        "scientific_configuration_not_operationally_tunable"
                    ),
                },
                {
                    "component": "htr_data_loader_workers",
                    "disposition": "unsupported_by_v1_executor",
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
            "disposition": "unsupported_by_v1_executor",
            "performance_claimed": False,
            "configured_candidate_values": [],
            "equality_gate": (
                "not_applicable_without_fit_row_keyed_cache_seam"
            ),
            "reason_code": (
                "fit_specific_tokenizer_is_retrained_and_no_authenticated_"
                "reusable_chunk_plan_api_exists"
            ),
            "component_dispositions": [
                {
                    "component": "cached_tokenizer",
                    "disposition": "unsupported_by_v1_executor",
                },
                {
                    "component": "cached_chunk_plan",
                    "disposition": "unsupported_by_v1_executor",
                },
            ],
            "code_evidence": evidence(
                "role_neutral_htr_group_execution",
                "production_role_neutral_producer_factories",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[4],
            "disposition": "unsupported_by_v1_executor",
            "performance_claimed": False,
            "configured_candidate_values": [],
            "equality_gate": (
                "not_applicable_without_multi_device_context_executor"
            ),
            "reason_code": (
                "v1_assigns_query_banks_to_single_devices_and_has_no_"
                "cross_device_context_primitive"
            ),
            "component_dispositions": [
                {
                    "component": (
                        "one_neural_query_context_per_accelerator_vs_"
                        "one_context_spanning_accelerators"
                    ),
                    "disposition": "unsupported_by_v1_executor",
                }
            ],
            "code_evidence": evidence(
                "role_neutral_neural_query_group_execution",
                "production_role_neutral_producer_factories",
            ),
        },
        {
            "axis_id": _REQUIRED_MATRIX_AXIS_IDS[5],
            "disposition": (
                "partially_measured"
                if compression["accepted"] is True
                else "equality_rejected"
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
            ],
            "equality_gate": (
                "compact_preflight_codecs_equal_lane_overlap_unmeasured"
                if compression["accepted"] is True
                else "compact_preflight_codec_scientific_equality_failed"
            ),
            "reason_code": (
                "compact_preflight_codec_choice_measured_but_complete_fits_"
                "still_lack_independent_cpu_gpu_lane_scheduling"
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
                    "disposition": "unsupported_by_v1_executor",
                    "performance_claimed": False,
                },
            ],
            "code_evidence": evidence(
                "role_neutral_performance_benchmark",
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
    child_peak_gpu_memory_bytes: int | None
    child_cpu_budget_attestation: Mapping[str, Any] | None
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


def _child_process_io_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[
    bool,
    int | None,
    int | None,
    float | None,
    float | None,
    int | None,
]:
    """Return one authenticated child-process delta for a one-owner fit."""

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
    telemetry = (
        owners[0].get("telemetry")
        if isinstance(owners[0], Mapping)
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
    return (
        True,
        int(process_io["read_bytes"]),
        int(process_io["write_bytes"]),
        float(wall),
        float(cpu),
        (
            None
            if peak_allocated is None and peak_reserved is None
            else max(int(peak_allocated or 0), int(peak_reserved or 0))
        ),
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
                int(instance.child_peak_gpu_memory_bytes or 0)
                for instance in instances
                if instance.device == device
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
    complete_ledger = TelemetryLedger(devices=(device,))
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
            resource_plan = ResourcePlan(
                devices=(device,),
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
        child_peak_gpu_memory_bytes,
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
        devices=(device,),
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
        child_peak_gpu_memory_bytes=child_peak_gpu_memory_bytes,
        child_cpu_budget_attestation=child_cpu_budget_attestation,
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
        "total_concurrency": candidate.total_concurrency,
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
    completed: Sequence[tuple[_InstanceResult, RoleNeutralBenchmarkWorkload]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = TelemetryLedger()
    rows: list[dict[str, Any]] = []
    with ledger.subphase(
        "benchmark.terminal_complete_artifact_audit",
        activity_kind="terminal_audit",
    ):
        for instance, workload in completed:
            if instance.manifest is None:
                continue
            validated = validate_role_neutral_stage1_execution(
                root=instance.root,
                plan=workload.plan,
            )
            if validated != instance.manifest:
                raise RuntimeError(
                    "terminal audit found a changed role-neutral execution"
                )
            tree_sha256, total_bytes, file_count = _audit_tree(
                instance.root,
                ledger=ledger,
            )
            rows.append(
                {
                    "root": str(instance.root),
                    "tree_sha256": tree_sha256,
                    "total_file_bytes": total_bytes,
                    "file_count": file_count,
                    "scientific_artifact_sha256": (
                        instance.scientific_artifact_sha256
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
) -> dict[str, Any]:
    """Run configured real role-neutral fits and publish measured selection."""

    if not isinstance(config, RoleNeutralBenchmarkConfig):
        raise TypeError("benchmark runner requires a typed config")
    destination = Path(output_root)
    if not destination.is_absolute():
        raise ValueError("benchmark output_root must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("benchmark output_root must be fresh")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError("benchmark output parent must be canonical")
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
    destination.mkdir(exist_ok=False)
    (destination / "warmups").mkdir(exist_ok=False)
    (destination / "runs").mkdir(exist_ok=False)
    (destination / "executor_sessions").mkdir(exist_ok=False)
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
    compression_benchmark = (
        run_compact_preflight_compression_benchmark(
            config=config.preflight_compression_benchmark,
            source=compression_sources[0],
            output_root=(
                destination / "preflight_compression_benchmark"
            ).resolve(),
        )
    )

    warmup_observations: list[BenchmarkRunObservation] = []
    warmup_details: list[dict[str, Any]] = []
    observations: list[BenchmarkRunObservation] = []
    details: list[dict[str, Any]] = []
    audit_targets: list[
        tuple[_InstanceResult, RoleNeutralBenchmarkWorkload]
    ] = []
    repetitions = (
        config.resource_performance_safety.minimum_benchmark_repetitions_per_scope
    )
    execution_rows: list[dict[str, Any]] = []
    sequence_index = 0
    candidates = tuple(config.candidates)
    executors: dict[tuple[str, str], Any] = {}

    def execute_observation(
        *,
        scope: RoleNeutralBenchmarkScope,
        candidate: RoleNeutralBenchmarkCandidate,
        observation_kind: str,
        observation_index: int,
        rotation_offset: int,
        candidate_position: int,
    ) -> None:
        nonlocal sequence_index
        key = (scope.label, candidate.name)
        workload = typed_workloads[scope.label]
        executor = executors.get(key)
        if executor is None:
            base_executor = workload.physical_owner_executor_builder(
                candidate.executor_mode,
                candidate.concurrency_per_device,
            )
            if candidate.executor_mode == "persistent_slots":
                open_session = getattr(base_executor, "open_session", None)
                if not callable(open_session):
                    raise TypeError(
                        "persistent benchmark candidate requires an executor "
                        "with open_session()"
                    )
                executor = open_session(
                    resources=_candidate_devices(
                        candidate=candidate,
                        inventory=resources,
                        safety=config.resource_performance_safety,
                    ),
                    max_workers=candidate.total_concurrency,
                    cpu_budget=candidate.host_cpu_budget,
                    marker_root=(
                        destination
                        / "executor_sessions"
                        / candidate.name
                        / scope.label
                    ).resolve(),
                )
            else:
                executor = base_executor
            executors[key] = executor
        execution_rows.append(
            {
                "sequence_index": sequence_index,
                "observation_kind": observation_kind,
                "scope_label": scope.label,
                "observation_index": observation_index,
                "rotation_offset": rotation_offset,
                "candidate_position": candidate_position,
                "candidate_name": candidate.name,
            }
        )
        observation, detail, instances = _run_observation(
            root=destination,
            config=config,
            candidate=candidate,
            scope=scope,
            workload=workload,
            repetition_index=observation_index,
            inventory=resources,
            physical_owner_executor=executor,
            observation_kind=observation_kind,
        )
        enriched_detail = {
            **detail,
            "execution_sequence_index": sequence_index,
            "candidate_position_within_rotation": candidate_position,
            "candidate_rotation_offset": rotation_offset,
        }
        if observation_kind == "warmup":
            warmup_observations.append(observation)
            warmup_details.append(enriched_detail)
        else:
            observations.append(observation)
            details.append(enriched_detail)
        audit_targets.extend(
            (instance, workload)
            for instance in instances
            if instance.manifest is not None
        )
        sequence_index += 1

    execution_failure: BaseException | None = None
    try:
        for scope_index, scope in enumerate(config.representative_scopes):
            for warmup_index in range(
                config.warmup_observations_per_candidate_scope
            ):
                rotation_offset = (
                    scope_index
                    * config.warmup_observations_per_candidate_scope
                    + warmup_index
                ) % len(candidates)
                rotated_candidates = (
                    candidates[rotation_offset:]
                    + candidates[:rotation_offset]
                )
                for candidate_position, candidate in enumerate(
                    rotated_candidates
                ):
                    execute_observation(
                        scope=scope,
                        candidate=candidate,
                        observation_kind="warmup",
                        observation_index=warmup_index,
                        rotation_offset=rotation_offset,
                        candidate_position=candidate_position,
                    )
            for repetition_index in range(repetitions):
                rotation_offset = (
                    scope_index * repetitions + repetition_index
                ) % len(candidates)
                rotated_candidates = (
                    candidates[rotation_offset:]
                    + candidates[:rotation_offset]
                )
                for candidate_position, candidate in enumerate(
                    rotated_candidates
                ):
                    execute_observation(
                        scope=scope,
                        candidate=candidate,
                        observation_kind="measured",
                        observation_index=repetition_index,
                        rotation_offset=rotation_offset,
                        candidate_position=candidate_position,
                    )
    except BaseException as exc:
        execution_failure = exc
    finally:
        close_failures: list[BaseException] = []
        for executor in executors.values():
            try:
                _close_physical_owner_executor(executor)
            except BaseException as exc:
                close_failures.append(exc)
        if execution_failure is None and close_failures:
            execution_failure = RuntimeError(
                "benchmark could not close an owned executor session"
            )
    if execution_failure is not None:
        raise execution_failure

    execution_schedule_body = {
        "schema_version": ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA,
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
        "entries": execution_rows,
    }
    execution_schedule = {
        **execution_schedule_body,
        "content_sha256": identity_sha256(execution_schedule_body),
    }

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
        candidate_warmup_rows = warmup_detail_by_candidate[candidate_name]
        expected_warmup_count = (
            len(config.representative_scopes)
            * config.warmup_observations_per_candidate_scope
        )
        warmup_telemetry_accepted = (
            len(candidate_warmup_rows) == expected_warmup_count
            and all(value["telemetry_accepted"] for value in candidate_warmup_rows)
        )
        warmup_matches_measured = warmup_telemetry_accepted and all(
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
        updated = {
            **row,
            "executor_mode": configured_candidate.executor_mode,
            "measured_observation_telemetry_accepted": telemetry_accepted,
            "warmup_observation_telemetry_accepted": (
                warmup_telemetry_accepted
            ),
            "warmup_scientific_identity_matches_measured": (
                warmup_matches_measured
            ),
            "accepted": (
                bool(row["accepted"])
                and telemetry_accepted
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
                int(value["execution_device_count"])
                * int(value["concurrency_per_device"]),
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
            "fastest_end_to_end_then_lower_total_concurrency_then_name_v1"
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
