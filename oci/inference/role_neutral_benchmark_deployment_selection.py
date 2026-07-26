"""Bind an accepted measured benchmark choice into a fresh deployment profile."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from .prepared_stage1_context import (
    PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
    load_prepared_stage1_context,
)
from .compact_preflight_compression_benchmark import (
    validate_compact_preflight_compression_benchmark_result,
)
from .portable_workflow_spec import (
    DeploymentProfile,
    ResourcePerformanceSafetyPolicy,
    ScientificWorkflowSpec,
    Stage1ExecutionProfile,
    identity_sha256,
)
from .production_text_preparation import stable_file_sha256
from .production_role_neutral_process_executor import _option_mapping
from .role_neutral_benchmark_workload_provider import (
    RoleNeutralBenchmarkWorkloadDeployment,
    _authenticate_paused_stage1_preflight,
    _stage1_build_options,
)
from .role_neutral_performance_benchmark import (
    ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA,
    ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA,
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkSourceBinding,
    build_role_neutral_benchmark_matrix_coverage,
)
from .role_neutral_performance_benchmark_publication import (
    ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST,
    RoleNeutralBenchmarkSelectionEvidence,
    load_role_neutral_benchmark_selection_evidence,
)
from .stage1_execution_topology_policy import (
    Stage1ExecutionTopologyPolicy,
)
from .stage1_htr_operational_controls import (
    RoleNeutralHTROperationalControls,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "config",
        "config_sha256",
        "workload_binding",
        "resource_inventory",
        "execution_schedule",
        "warmup_observations",
        "warmup_telemetry",
        "warmup_observations_excluded_from_selection",
        "benchmark_observations",
        "observation_telemetry",
        "terminal_audit",
        "terminal_audit_telemetry",
        "ordinary_observations_exclude_terminal_audit",
        "candidate_results",
        "preflight_compression_benchmark",
        "benchmark_matrix_coverage",
        "selected_candidate",
        "selection_policy",
        "scientific_result_identity_sha256",
        "accepted",
        "content_sha256",
    }
)
_PATH_FIELDS = (
    "dataset_path",
    "durable_artifact_root",
    "scratch_root",
    "embedding_model_locator",
    "htr_model_locator",
    "stage1_profile_locator",
    "query_profile_locator",
    "stage2_tokenizer_locator",
    "oracle_source",
)


@dataclass(frozen=True)
class _AuthenticatedBenchmarkEvidence:
    kind: str
    result: Mapping[str, Any]
    result_file_sha256: str
    result_content_sha256: str
    config: RoleNeutralBenchmarkConfig
    workload_binding: Mapping[str, Any]
    source_binding: RoleNeutralBenchmarkSourceBinding
    scientific_workflow_binding: Mapping[str, Any] | None = None
    raw_result_locator: Path | None = None
    publication_manifest_locator: Path | None = None
    publication_manifest_file_sha256: str | None = None
    publication_path_neutral_content_root_sha256: str | None = None


def _strict_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"benchmark result contains duplicate key {key!r}")
        output[key] = value
    return output


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


def _read_result(
    path: Path,
) -> tuple[dict[str, Any], str, RoleNeutralBenchmarkConfig]:
    state = os.lstat(path)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISREG(state.st_mode)
        or int(state.st_nlink) != 1
    ):
        raise ValueError("benchmark result must be one private regular file")
    source = path.resolve(strict=True)
    if Path(os.path.abspath(os.fspath(path))) != source:
        raise ValueError("benchmark result parent path must be symlink-free")
    file_sha256, _size = stable_file_sha256(source)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"benchmark result contains {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark result is not closed UTF-8 JSON") from exc
    if _stat_identity(state) != _stat_identity(os.lstat(source)):
        raise RuntimeError("benchmark result changed while it was being read")
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        raise ValueError("benchmark result does not match its closed schema")
    if value["schema_version"] != ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA:
        raise ValueError("unsupported benchmark result schema")
    supplied_content = value["content_sha256"]
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if supplied_content != identity_sha256(body):
        raise ValueError("benchmark result content identity is invalid")
    if (
        not isinstance(value["config"], Mapping)
        or value["config_sha256"] != identity_sha256(value["config"])
    ):
        raise ValueError("benchmark result config identity is invalid")
    config = RoleNeutralBenchmarkConfig.from_mapping(value["config"])
    _validate_execution_schedule(
        value["execution_schedule"],
        config=config,
    )
    candidate_results = value["candidate_results"]
    compression_benchmark = (
        validate_compact_preflight_compression_benchmark_result(
            value["preflight_compression_benchmark"],
            reopen_artifacts=True,
        )
    )
    if (
        not isinstance(candidate_results, list)
        or compression_benchmark["config"]
        != config.preflight_compression_benchmark.as_dict()
        or value["benchmark_matrix_coverage"]
        != build_role_neutral_benchmark_matrix_coverage(
            config=config,
            candidate_rows=candidate_results,
            compression_benchmark_result=compression_benchmark,
        )
    ):
        raise ValueError(
            "benchmark result matrix-axis coverage is incomplete or changed"
        )
    if (
        value["status"] != "complete"
        or value["accepted"] is not True
        or value["ordinary_observations_exclude_terminal_audit"] is not True
        or value["warmup_observations_excluded_from_selection"] is not True
        or value["selection_policy"]
        != "fastest_end_to_end_then_lower_effective_owner_concurrency_then_name_v2"
        or _SHA256.fullmatch(
            str(value["scientific_result_identity_sha256"])
        )
        is None
    ):
        raise ValueError("benchmark result is not an accepted complete measurement")
    terminal = value["terminal_audit"]
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("exactly_one_completed_terminal_audit") is not True
    ):
        raise ValueError("benchmark result lacks its complete terminal audit")
    return value, file_sha256, config


def _publication_root_from_locator(path: Path) -> Path | None:
    state = os.lstat(path)
    absolute = Path(os.path.abspath(os.fspath(path)))
    if stat.S_ISLNK(state.st_mode):
        raise ValueError(
            "benchmark evidence locator must be one private regular file or "
            "a canonical publication root; symlinks are forbidden"
        )
    if stat.S_ISDIR(state.st_mode):
        root = path.resolve(strict=True)
        if root != absolute:
            raise ValueError(
                "benchmark publication root must be canonical and symlink-free"
            )
        return root
    if path.name != ROLE_NEUTRAL_BENCHMARK_PUBLICATION_MANIFEST:
        return None
    if not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
        raise ValueError(
            "benchmark publication manifest must be one private regular file"
        )
    manifest = path.resolve(strict=True)
    if manifest != absolute:
        raise ValueError(
            "benchmark publication manifest path must be canonical and symlink-free"
        )
    return manifest.parent


def _validated_publication_result(
    evidence: RoleNeutralBenchmarkSelectionEvidence,
) -> tuple[dict[str, Any], RoleNeutralBenchmarkConfig]:
    result = dict(evidence.normalized_benchmark_result)
    if (
        result.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("accepted") is not True
        or result.get("ordinary_observations_exclude_terminal_audit")
        is not True
        or result.get("warmup_observations_excluded_from_selection")
        is not True
        or result.get("selection_policy")
        != "fastest_end_to_end_then_lower_effective_owner_concurrency_then_name_v2"
        or not isinstance(result.get("config"), Mapping)
        or result.get("config_sha256")
        != identity_sha256(result["config"])
        or _SHA256.fullmatch(
            str(result.get("scientific_result_identity_sha256"))
        )
        is None
    ):
        raise ValueError(
            "published path-neutral benchmark result is not accepted"
        )
    config = RoleNeutralBenchmarkConfig.from_mapping(result["config"])
    _validate_execution_schedule(result["execution_schedule"], config=config)
    matrix = result.get("benchmark_matrix_coverage")
    compression = result.get("preflight_compression_benchmark")
    if (
        not isinstance(matrix, Mapping)
        or matrix.get("all_required_axes_accounted") is not True
        or not isinstance(compression, Mapping)
        or compression.get("accepted") is not True
        or not isinstance(
            compression.get("selected_parquet_compression"),
            str,
        )
    ):
        raise ValueError(
            "published path-neutral benchmark coverage is incomplete"
        )
    return result, config


def _read_benchmark_evidence(
    path: Path,
) -> _AuthenticatedBenchmarkEvidence:
    publication_root = _publication_root_from_locator(path)
    if publication_root is None:
        result, file_sha256, config = _read_result(path)
        source = _validate_workload_binding(
            value=result["workload_binding"],
            config=config,
        )
        return _AuthenticatedBenchmarkEvidence(
            kind="raw_result_v1",
            result=result,
            result_file_sha256=file_sha256,
            result_content_sha256=str(result["content_sha256"]),
            config=config,
            workload_binding=result["workload_binding"],
            source_binding=source,
            raw_result_locator=path.resolve(strict=True),
        )

    published = load_role_neutral_benchmark_selection_evidence(
        publication_root
    )
    result, config = _validated_publication_result(published)
    source = _validate_workload_binding(
        value=published.workload_binding,
        config=config,
    )
    if source != published.source_binding:
        raise ValueError(
            "published benchmark workload source binding changed"
        )
    return _AuthenticatedBenchmarkEvidence(
        kind="durable_publication_v1",
        result=result,
        result_file_sha256=published.benchmark_result_file_sha256,
        result_content_sha256=published.benchmark_result_content_sha256,
        config=config,
        workload_binding=published.workload_binding,
        source_binding=source,
        scientific_workflow_binding=(
            published.scientific_workflow_binding
        ),
        publication_manifest_locator=(
            published.publication_manifest_path
        ),
        publication_manifest_file_sha256=(
            published.publication_manifest_file_sha256
        ),
        publication_path_neutral_content_root_sha256=(
            published.publication_manifest.path_neutral_content_root_sha256
        ),
    )


def _authenticate_published_scientific_spec(
    *,
    scientific_spec_path: Path,
    binding: Mapping[str, Any],
) -> tuple[ScientificWorkflowSpec, str]:
    scientific_source = scientific_spec_path.resolve(strict=True)
    scientific_spec_file_sha256, _scientific_size = stable_file_sha256(
        scientific_source
    )
    scientific_spec = ScientificWorkflowSpec.from_json(scientific_source)
    portable_identity = scientific_spec.identity_payload()
    if (
        binding.get("scientific_spec_source_sha256")
        != scientific_spec_file_sha256
        or binding.get("portable_scientific_spec")
        != portable_identity
        or binding.get("portable_scientific_spec_sha256")
        != identity_sha256(portable_identity)
    ):
        raise ValueError(
            "supplied scientific spec differs from the benchmarked workflow"
        )
    return scientific_spec, scientific_spec_file_sha256


def _validate_execution_schedule(
    value: Any,
    *,
    config: RoleNeutralBenchmarkConfig,
) -> None:
    required = {
        "schema_version",
        "warmup_policy",
        "warmup_observations_per_candidate_scope",
        "candidate_order_policy",
        "candidate_names_in_configured_order",
        "entries",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("benchmark execution schedule is not closed")
    body = {
        key: item for key, item in value.items() if key != "content_sha256"
    }
    if (
        value["schema_version"]
        != ROLE_NEUTRAL_BENCHMARK_EXECUTION_SCHEDULE_SCHEMA
        or value["warmup_policy"]
        != "configured_complete_observations_excluded_from_selection_v1"
        or value["warmup_observations_per_candidate_scope"]
        != config.warmup_observations_per_candidate_scope
        or value["candidate_order_policy"]
        != "scope_observation_latin_rotation_with_warmup_v2"
        or value["content_sha256"] != identity_sha256(body)
    ):
        raise ValueError("benchmark execution schedule identity is invalid")
    candidates = tuple(config.candidates)
    if value["candidate_names_in_configured_order"] != [
        candidate.name for candidate in candidates
    ]:
        raise ValueError("benchmark execution schedule changed candidate order")
    expected: list[dict[str, Any]] = []
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
                expected.append(
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
                expected.append(
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
    if value["entries"] != expected:
        raise ValueError("benchmark execution schedule was not counterbalanced")


def _validate_workload_binding(
    *,
    value: Any,
    config: RoleNeutralBenchmarkConfig,
) -> RoleNeutralBenchmarkSourceBinding:
    required = {
        "schema_version",
        "source",
        "representative_scope_plans",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("benchmark workload binding does not match its closed schema")
    if value["schema_version"] != ROLE_NEUTRAL_BENCHMARK_WORKLOAD_BINDING_SCHEMA:
        raise ValueError("unsupported benchmark workload-binding schema")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value["content_sha256"] != identity_sha256(body):
        raise ValueError("benchmark workload-binding content identity is invalid")
    raw_source = value["source"]
    source_fields = {
        "schema_version",
        "workflow_request_sha256",
        "workflow_scientific_sha256",
        "workload_deployment_sha256",
        "stage1_preflight_phase_content_sha256",
        "prepared_stage1_context_content_root_sha256",
    }
    if not isinstance(raw_source, Mapping) or set(raw_source) != source_fields:
        raise ValueError("benchmark source binding does not match its closed schema")
    if (
        raw_source.get("schema_version")
        != ROLE_NEUTRAL_BENCHMARK_SOURCE_BINDING_SCHEMA
    ):
        raise ValueError("unsupported benchmark source-binding schema")
    source = RoleNeutralBenchmarkSourceBinding(**dict(raw_source))
    raw_plans = value["representative_scope_plans"]
    if not isinstance(raw_plans, list):
        raise ValueError("benchmark workload plans must be a list")
    expected_scopes = list(config.representative_scopes)
    if len(raw_plans) != len(expected_scopes):
        raise ValueError("benchmark workload plans do not match configured scopes")
    for row, expected in zip(raw_plans, expected_scopes):
        required_plan = {
            "scope_label",
            "fit_row_count",
            "plan_scientific_content_sha256",
            "physical_owner_scope_id",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != required_plan
            or row.get("scope_label") != expected.label
            or row.get("fit_row_count") != expected.fit_row_count
            or _SHA256.fullmatch(
                str(row.get("plan_scientific_content_sha256", ""))
            )
            is None
            or not isinstance(row.get("physical_owner_scope_id"), str)
            or not row["physical_owner_scope_id"].strip()
        ):
            raise ValueError(
                "benchmark workload plan differs from its configured scope"
            )
    return source


def _read_workload_deployment(
    path: Path,
) -> tuple[RoleNeutralBenchmarkWorkloadDeployment, str]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError("workload deployment must be one private regular file")
    source = path.resolve(strict=True)
    if Path(os.path.abspath(os.fspath(path))) != source:
        raise ValueError("workload deployment parent path must be symlink-free")
    digest, _size = stable_file_sha256(source)
    deployment = RoleNeutralBenchmarkWorkloadDeployment.from_json(source)
    if _stat_identity(before) != _stat_identity(os.lstat(source)):
        raise RuntimeError("workload deployment changed while it was being read")
    return deployment, digest


def _authenticate_prepared_context_binding(
    *,
    deployment: RoleNeutralBenchmarkWorkloadDeployment,
    authenticated: Any,
    expected_content_root_sha256: str,
) -> Mapping[str, Any]:
    """Reopen the exact prepared context and compare every producer input."""

    manifest = (
        deployment.prepared_context_root
        / "sealed_prepared_stage1_context"
        / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    )
    artifact = load_prepared_stage1_context(manifest)
    expected_root = (
        deployment.prepared_context_root
        / "sealed_prepared_stage1_context"
    ).resolve(strict=True)
    request = authenticated.request
    portable = request.get("portable_scientific_spec")
    architecture_profiles = (
        portable.get("architecture_profiles")
        if isinstance(portable, Mapping)
        else None
    )
    expected_options = _option_mapping(
        SimpleNamespace(
            options=_stage1_build_options(
                authenticated=authenticated,
                deployment=deployment,
            )
        )
    )
    locators = artifact.execution_locators
    if (
        artifact.root != expected_root
        or artifact.content_root_sha256 != expected_content_root_sha256
        or not isinstance(architecture_profiles, Mapping)
        or locators.get("architecture_profiles") != architecture_profiles
        or locators.get("runtime_compatibility_class")
        != request.get("runtime_compatibility_class")
        or locators.get("stage1_build_options") != expected_options
        or expected_options.get("portable_cluster_preflight_v2") is not True
    ):
        raise ValueError(
            "prepared Stage 1 context differs from the benchmark source"
        )
    return {
        "manifest_path": str(artifact.manifest_path),
        "content_root_sha256": artifact.content_root_sha256,
        "scientific_compatibility_sha256": (
            artifact.scientific_compatibility_sha256
        ),
    }


def _authenticate_selection_source(
    *,
    workload_binding: Mapping[str, Any],
    config: RoleNeutralBenchmarkConfig,
    workload_deployment_path: Path,
    scientific_spec_path: Path,
) -> RoleNeutralBenchmarkSourceBinding:
    source = _validate_workload_binding(
        value=workload_binding,
        config=config,
    )
    deployment, deployment_sha256 = _read_workload_deployment(
        workload_deployment_path
    )
    if deployment_sha256 != source.workload_deployment_sha256:
        raise ValueError("benchmark result names a different workload deployment")
    if (
        deployment.expected_benchmark_config_sha256
        != identity_sha256(config.as_dict())
    ):
        raise ValueError("workload deployment names a different benchmark config")
    authenticated = _authenticate_paused_stage1_preflight(
        deployment,
        require_fresh_prepared_context=False,
    )
    _authenticate_prepared_context_binding(
        deployment=deployment,
        authenticated=authenticated,
        expected_content_root_sha256=(
            source.prepared_stage1_context_content_root_sha256
        ),
    )
    request = authenticated.request
    if (
        deployment.expected_workflow_request_sha256
        != source.workflow_request_sha256
        or request.get("request_sha256") != source.workflow_request_sha256
    ):
        raise ValueError("benchmark result names a different staged workflow request")
    scientific_identity = request.get("scientific_identity")
    if (
        not isinstance(scientific_identity, Mapping)
        or scientific_identity.get("scientific_sha256")
        != source.workflow_scientific_sha256
    ):
        raise ValueError("benchmark result names a different staged scientific identity")
    preflight = authenticated.phases.get("stage1_preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("content_sha256")
        != source.stage1_preflight_phase_content_sha256
    ):
        raise ValueError("benchmark result names a different staged preflight")

    scientific_source = scientific_spec_path.resolve(strict=True)
    scientific_sha256, _size = stable_file_sha256(scientific_source)
    scientific = ScientificWorkflowSpec.from_json(scientific_source)
    if request.get("portable_scientific_spec") != scientific.identity_payload():
        raise ValueError(
            "staged workflow scientific settings differ from the supplied spec"
        )
    expected_source_sha256 = request.get("scientific_spec_source_sha256")
    if (
        not isinstance(expected_source_sha256, str)
        or expected_source_sha256 != scientific_sha256
    ):
        raise ValueError(
            "supplied scientific spec is not the immutable staged source"
        )
    return source


def _selected_candidate(
    result: Mapping[str, Any],
    *,
    config: RoleNeutralBenchmarkConfig,
) -> tuple[str, Mapping[str, Any]]:
    selected = result.get("selected_candidate")
    result_rows = result.get("candidate_results")
    if (
        not isinstance(selected, str)
        or not selected.strip()
        or not isinstance(result_rows, list)
    ):
        raise ValueError("benchmark result lacks a selected configured candidate")
    configured_by_name = {value.name: value for value in config.candidates}
    configured = configured_by_name.get(selected)
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    for row in result_rows:
        if not isinstance(row, Mapping):
            raise ValueError("benchmark candidate results contain an invalid row")
        name = row.get("candidate_name")
        if (
            not isinstance(name, str)
            or name not in configured_by_name
            or name in rows_by_name
        ):
            raise ValueError("benchmark candidate results are not one-to-one")
        rows_by_name[name] = row
    if set(rows_by_name) != set(configured_by_name) or configured is None:
        raise ValueError("benchmark candidate results do not cover the config")
    accepted: list[tuple[Mapping[str, Any], Any]] = []
    for name, row in rows_by_name.items():
        candidate = configured_by_name[name]
        if row.get("accepted") is not True:
            continue
        if (
            row.get("measured_observation_telemetry_accepted") is not True
            or row.get("warmup_observation_telemetry_accepted") is not True
            or row.get("warmup_scientific_identity_matches_measured") is not True
            or row.get("executor_mode") != candidate.executor_mode
            or row.get("execution_device_count")
            != (candidate.accelerator_count or 1)
            or row.get("concurrency_per_device")
            != candidate.concurrency_per_device
            or row.get("effective_parallel_owners")
            != candidate.total_concurrency
            or row.get("resource_slot_count")
            != candidate.resource_slot_count
            or row.get("neural_query_topology")
            != candidate.neural_query_topology.as_dict()
            or row.get("htr_operational_controls")
            != candidate.htr_operational_controls.as_dict()
            or row.get("cpu_gpu_lane_interval_telemetry_accepted")
            is not True
            or isinstance(
                row.get("cpu_gpu_lane_overlap_observation_count"),
                bool,
            )
            or not isinstance(
                row.get("cpu_gpu_lane_overlap_observation_count"),
                int,
            )
            or int(
                row["cpu_gpu_lane_overlap_observation_count"]
            )
            < (1 if candidate.accelerator_count > 0 else 0)
            or row.get("cpu_gpu_lane_overlap_descriptive_only")
            is not True
            or row.get("cpu_gpu_lane_overlap_speedup_claimed")
            is not False
        ):
            raise ValueError("accepted benchmark candidate result is inconsistent")
        throughput = row.get("throughput_fit_rows_per_second")
        if (
            isinstance(throughput, bool)
            or not isinstance(throughput, (int, float))
            or float(throughput) <= 0
        ):
            raise ValueError("accepted benchmark candidate lacks valid throughput")
        accepted.append((row, candidate))
    expected_selected = (
        None
        if not accepted
        else min(
            accepted,
            key=lambda item: (
                -float(item[0]["throughput_fit_rows_per_second"]),
                int(item[0]["effective_parallel_owners"]),
                str(item[0]["candidate_name"]),
            ),
        )[1].name
    )
    if expected_selected != selected:
        raise ValueError("selected benchmark candidate violates selection policy")
    selected_rows = [
        row
        for row, candidate in accepted
        if candidate.name == selected
    ]
    if len(selected_rows) != 1:
        raise ValueError("selected benchmark candidate is not uniquely accepted")
    return selected.strip(), asdict(configured)


def _selected_devices(
    *,
    base: DeploymentProfile,
    accelerator_count: int,
) -> tuple[tuple[str, ...], int]:
    if isinstance(accelerator_count, bool) or not isinstance(accelerator_count, int):
        raise ValueError("selected candidate accelerator_count is invalid")
    if accelerator_count < 0:
        raise ValueError("selected candidate accelerator_count is invalid")
    if accelerator_count == 0:
        return ("cpu",), 1
    if base.devices == ("cpu",):
        raise ValueError("accelerator benchmark selection conflicts with CPU deployment")
    if base.devices == ("auto",):
        return base.devices, accelerator_count
    if len(base.devices) < accelerator_count:
        raise ValueError(
            "benchmark selection requires more accelerators than the deployment "
            "profile permits"
        )
    return tuple(base.devices[:accelerator_count]), accelerator_count


def _write_profile(path: Path, profile: DeploymentProfile) -> None:
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("selected deployment profile path must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("selected deployment profile path must be fresh")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError("selected deployment profile parent must be canonical")
    payload = (
        json.dumps(
            asdict(profile),
            default=str,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        destination,
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
                raise OSError("selected deployment profile write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _rebase_profile_paths(
    profile: DeploymentProfile,
    *,
    source_path: Path,
) -> DeploymentProfile:
    source_parent = source_path.resolve(strict=True).parent
    replacements: dict[str, Path | None] = {}
    for field_name in _PATH_FIELDS:
        value = getattr(profile, field_name)
        if value is None:
            replacements[field_name] = None
            continue
        path = Path(value)
        if not path.is_absolute():
            path = source_parent / path
        replacements[field_name] = path.resolve(strict=False)
    return replace(profile, **replacements)


def validate_benchmarked_stage1_execution_profile(
    *,
    profile: Stage1ExecutionProfile,
    scientific_spec_path: Path | str,
    resource_performance_safety: ResourcePerformanceSafetyPolicy,
    cpu_budget: int,
) -> Mapping[str, Any]:
    """Freshly reopen every measured-selection authority.

    This validator is intentionally suitable for initial compilation, resume,
    and terminal validation.  A stored digest alone is never accepted: both
    exact evidence files and the staged workflow request/preflight are
    authenticated again.
    """

    if not isinstance(profile, Stage1ExecutionProfile):
        raise TypeError("benchmark evidence validation requires a typed profile")
    if profile.selection_method != "measured_role_neutral_benchmark_v1":
        raise ValueError("Stage 1 execution profile is not benchmark-selected")
    if not isinstance(
        resource_performance_safety,
        ResourcePerformanceSafetyPolicy,
    ):
        raise TypeError("benchmark evidence validation requires typed safety")
    if isinstance(cpu_budget, bool) or not isinstance(cpu_budget, int) or cpu_budget < 1:
        raise ValueError("benchmark evidence validation requires a positive CPU budget")
    assert profile.benchmark_result_sha256 is not None
    assert profile.benchmark_workload_deployment_sha256 is not None
    if profile.benchmark_evidence_kind == "raw_result_v1":
        assert profile.benchmark_result_locator is not None
        evidence_locator = profile.benchmark_result_locator
    elif profile.benchmark_evidence_kind == "durable_publication_v1":
        assert profile.benchmark_publication_locator is not None
        evidence_locator = profile.benchmark_publication_locator
    else:
        raise ValueError(
            "Stage 1 execution profile has unsupported benchmark evidence"
        )
    evidence = _read_benchmark_evidence(evidence_locator)
    if evidence.kind != profile.benchmark_evidence_kind:
        raise ValueError(
            "benchmark evidence kind differs from the selected profile"
        )
    if evidence.result_file_sha256 != profile.benchmark_result_sha256:
        raise ValueError("benchmark result differs from the selected profile")
    if evidence.kind == "raw_result_v1":
        assert profile.benchmark_workload_deployment_locator is not None
        source = _authenticate_selection_source(
            workload_binding=evidence.workload_binding,
            config=evidence.config,
            workload_deployment_path=(
                profile.benchmark_workload_deployment_locator
            ),
            scientific_spec_path=Path(scientific_spec_path),
        )
    else:
        assert profile.benchmark_publication_sha256 is not None
        if (
            evidence.publication_manifest_file_sha256
            != profile.benchmark_publication_sha256
            or evidence.publication_manifest_locator
            != profile.benchmark_publication_locator
        ):
            raise ValueError(
                "benchmark publication differs from the selected profile"
            )
        source = evidence.source_binding
        if not isinstance(
            evidence.scientific_workflow_binding,
            Mapping,
        ):
            raise ValueError(
                "benchmark publication lacks its scientific workflow binding"
            )
        scientific_spec, scientific_spec_file_sha256 = (
            _authenticate_published_scientific_spec(
                scientific_spec_path=Path(scientific_spec_path),
                binding=evidence.scientific_workflow_binding,
            )
        )
    if (
        source.workload_deployment_sha256
        != profile.benchmark_workload_deployment_sha256
    ):
        raise ValueError(
            "benchmark workload deployment differs from the selected profile"
        )
    result = evidence.result
    config = evidence.config
    if config.resource_performance_safety != resource_performance_safety:
        raise ValueError(
            "benchmark safety policy differs from the selected deployment"
        )
    selected_name, candidate = _selected_candidate(result, config=config)
    accelerator_count = int(candidate["accelerator_count"])
    selected_topology = Stage1ExecutionTopologyPolicy.from_mapping(
        candidate["neural_query_topology"]
    )
    selected_htr_controls = (
        RoleNeutralHTROperationalControls.from_mapping(
            candidate["htr_operational_controls"]
        )
    )
    expected_resource_kind = (
        "cpu" if accelerator_count == 0 else "accelerator"
    )
    expected_device_count = accelerator_count or 1
    expected_parallel_owners = (
        selected_topology.effective_parallel_owners_for_shape(
            resource_kind=expected_resource_kind,
            device_count=expected_device_count,
            workers_per_device=int(
                candidate["concurrency_per_device"]
            ),
        )
    )
    if (
        selected_name != profile.selected_candidate
        or expected_resource_kind != profile.resource_kind
        or expected_device_count != profile.device_count
        or int(candidate["concurrency_per_device"])
        != profile.scope_workers_per_device
        or expected_parallel_owners != profile.max_parallel_owners
        or str(candidate["executor_mode"]) != profile.executor_mode
        or selected_topology != profile.neural_query_topology
        or selected_htr_controls != profile.htr_operational_controls
        or int(candidate["host_cpu_budget"]) != int(cpu_budget)
    ):
        raise ValueError(
            "selected Stage 1 execution profile differs from its benchmark"
        )
    body = {
        "schema_version": (
            "portable_benchmarked_stage1_execution_revalidation_v4"
        ),
        "selected_candidate": selected_name,
        "benchmark_evidence_kind": evidence.kind,
        "benchmark_result_sha256": evidence.result_file_sha256,
        "benchmark_result_content_sha256": (
            evidence.result_content_sha256
        ),
        "benchmark_publication_sha256": (
            evidence.publication_manifest_file_sha256
        ),
        "benchmark_publication_path_neutral_content_root_sha256": (
            evidence.publication_path_neutral_content_root_sha256
        ),
        "benchmark_workload_deployment_sha256": (
            source.workload_deployment_sha256
        ),
        "workflow_request_sha256": source.workflow_request_sha256,
        "workflow_scientific_sha256": source.workflow_scientific_sha256,
        "stage1_preflight_phase_content_sha256": (
            source.stage1_preflight_phase_content_sha256
        ),
        "prepared_stage1_context_content_root_sha256": (
            source.prepared_stage1_context_content_root_sha256
        ),
        "resource_kind": expected_resource_kind,
        "device_count": accelerator_count or 1,
        "scope_workers_per_device": int(candidate["concurrency_per_device"]),
        "max_parallel_owners": expected_parallel_owners,
        "executor_mode": str(candidate["executor_mode"]),
        "neural_query_topology": selected_topology.as_dict(),
        "htr_operational_controls": selected_htr_controls.as_dict(),
        "cpu_budget": int(cpu_budget),
        "resource_performance_safety_sha256": (
            resource_performance_safety.content_sha256
        ),
        "scientific_spec_file_sha256": (
            None
            if evidence.kind == "raw_result_v1"
            else scientific_spec_file_sha256
        ),
        "scientific_spec_scientific_sha256": (
            None
            if evidence.kind == "raw_result_v1"
            else scientific_spec.scientific_sha256
        ),
        "benchmark_matrix_coverage": result[
            "benchmark_matrix_coverage"
        ],
        "selected_preflight_parquet_compression": result[
            "preflight_compression_benchmark"
        ]["selected_parquet_compression"],
    }
    return {**body, "content_sha256": identity_sha256(body)}


def select_benchmarked_deployment_profile(
    *,
    base_deployment_path: Path | str,
    benchmark_result_path: Path | str | None = None,
    benchmark_publication_path: Path | str | None = None,
    benchmark_workload_deployment_path: Path | str | None = None,
    scientific_spec_path: Path | str,
    output_path: Path | str,
) -> DeploymentProfile:
    """Authenticate a benchmark and publish its selected operational profile."""

    if (benchmark_result_path is None) == (
        benchmark_publication_path is None
    ):
        raise ValueError(
            "select exactly one raw benchmark result or durable publication"
        )
    base_source = Path(base_deployment_path).resolve(strict=True)
    base = _rebase_profile_paths(
        DeploymentProfile.from_json(base_source),
        source_path=base_source,
    )
    if base.stage1_execution.selection_method != "operator_configured":
        raise ValueError("base deployment already claims measured benchmark selection")
    evidence_path = Path(
        benchmark_publication_path
        if benchmark_publication_path is not None
        else benchmark_result_path
    )
    evidence = _read_benchmark_evidence(evidence_path)
    expected_evidence_kind = (
        "durable_publication_v1"
        if benchmark_publication_path is not None
        else "raw_result_v1"
    )
    if evidence.kind != expected_evidence_kind:
        raise ValueError(
            "benchmark evidence input kind differs from its explicit argument"
        )
    result = evidence.result
    config = evidence.config
    if evidence.kind == "raw_result_v1":
        if benchmark_workload_deployment_path is None:
            raise ValueError(
                "raw benchmark selection requires its workload deployment"
            )
        source_binding = _authenticate_selection_source(
            workload_binding=evidence.workload_binding,
            config=config,
            workload_deployment_path=Path(
                benchmark_workload_deployment_path
            ),
            scientific_spec_path=Path(scientific_spec_path),
        )
    else:
        if benchmark_workload_deployment_path is not None:
            raise ValueError(
                "durable benchmark selection forbids a historical workload "
                "deployment locator"
            )
        source_binding = evidence.source_binding
        if not isinstance(
            evidence.scientific_workflow_binding,
            Mapping,
        ):
            raise ValueError(
                "benchmark publication lacks its scientific workflow binding"
            )
        _authenticate_published_scientific_spec(
            scientific_spec_path=Path(scientific_spec_path),
            binding=evidence.scientific_workflow_binding,
        )
    if config.resource_performance_safety != base.resource_performance_safety:
        raise ValueError(
            "benchmark safety policy differs from the target deployment"
        )
    selected_name, candidate = _selected_candidate(result, config=config)
    if candidate["host_cpu_budget"] != base.cpu_budget:
        raise ValueError(
            "selected benchmark CPU budget differs from the target deployment"
        )
    accelerator_count = candidate["accelerator_count"]
    concurrency = candidate["concurrency_per_device"]
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("selected candidate concurrency_per_device is invalid")
    topology = Stage1ExecutionTopologyPolicy.from_mapping(
        candidate["neural_query_topology"]
    )
    htr_controls = RoleNeutralHTROperationalControls.from_mapping(
        candidate["htr_operational_controls"]
    )
    devices, execution_device_count = _selected_devices(
        base=base,
        accelerator_count=accelerator_count,
    )
    selected_profile = replace(
        base,
        devices=devices,
        cluster_preflight_parquet_compression=result[
            "preflight_compression_benchmark"
        ]["selected_parquet_compression"],
        stage1_execution=Stage1ExecutionProfile(
            resource_kind=(
                "cpu" if accelerator_count == 0 else "accelerator"
            ),
            device_count=execution_device_count,
            scope_workers_per_device=concurrency,
            max_parallel_owners=(
                topology.effective_parallel_owners_for_shape(
                    resource_kind=(
                        "cpu"
                        if accelerator_count == 0
                        else "accelerator"
                    ),
                    device_count=execution_device_count,
                    workers_per_device=concurrency,
                )
            ),
            executor_mode=str(candidate["executor_mode"]),
            persistent_slot_startup_timeout_seconds=(
                base.stage1_execution
                .persistent_slot_startup_timeout_seconds
            ),
            neural_query_topology=topology,
            htr_operational_controls=htr_controls,
            selection_method="measured_role_neutral_benchmark_v1",
            benchmark_evidence_kind=evidence.kind,
            selected_candidate=selected_name,
            benchmark_result_sha256=evidence.result_file_sha256,
            benchmark_result_locator=evidence.raw_result_locator,
            benchmark_workload_deployment_sha256=(
                source_binding.workload_deployment_sha256
            ),
            benchmark_workload_deployment_locator=(
                Path(benchmark_workload_deployment_path).resolve(strict=True)
                if evidence.kind == "raw_result_v1"
                else None
            ),
            benchmark_publication_sha256=(
                evidence.publication_manifest_file_sha256
            ),
            benchmark_publication_locator=(
                evidence.publication_manifest_locator
            ),
        ),
    )
    destination = Path(output_path)
    _write_profile(destination, selected_profile)
    reopened = DeploymentProfile.from_json(destination)
    if reopened != selected_profile:
        raise RuntimeError("selected deployment profile changed during publication")
    return reopened


__all__ = [
    "select_benchmarked_deployment_profile",
    "validate_benchmarked_stage1_execution_profile",
]
