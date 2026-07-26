"""Authenticated real-workload provider for the role-neutral benchmark.

The provider consumes a closed operational deployment document and an already
sealed workflow paused immediately after ``stage1_preflight``.  It does not
discover cohort sizes, paths, columns, devices, or models from source
constants.  Instead, it reopens the immutable workflow request and its three
completed phase trees, reconstructs the exact prepared Stage 1 context, and
selects representative one-owner plans by configured logical purpose and
fit-row content.

No model is fitted here.  The returned callables bind the authenticated
prepared context to ``PreparedBuildRoleNeutralProducerFactoriesBuilder``; the
benchmark runner remains solely responsible for executing measured fits.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .performance_telemetry import ImmutableInputObservation
from .prepared_stage1_context import (
    PREPARED_STAGE1_CONTEXT_MANIFEST_NAME,
    PreparedStage1ContextArtifact,
    load_prepared_stage1_context,
    seal_prepared_stage1_context,
    serialize_stage1_build_options,
)
from .portable_workflow_spec import (
    SentenceEmbeddingEncoderSpec,
    Stage1ExecutionProfile,
    identity_sha256,
)
from .production_all_evidence_workflow import (
    EMBEDDING_CACHE_PHASE_SCHEMA,
    STAGE1_PREFLIGHT_PHASE_SCHEMA,
    WORKFLOW_PROGRESS_SCHEMA,
    WORKFLOW_SCHEMA,
    _read_json_object,
    _revalidate_request_bound_external_inputs,
    _sha,
    _validate_phase_manifest_from_paths,
)
from .production_role_neutral_producer_factories import (
    PreparedBuildRoleNeutralProducerFactoriesBuilder,
)
from .production_role_neutral_process_executor import (
    ProcessIsolatedRoleNeutralPhysicalOwnerExecutor,
)
from .production_role_neutral_persistent_executor import (
    PersistentSpawnRoleNeutralPhysicalOwnerExecutor,
)
from .production_stage1_bundle import (
    ProductionStage1BundleBuilder,
    Stage1BundleBuildOptions,
)
from .production_stage1_scope_scheduler import (
    Stage1PhysicalFitIdentity,
    Stage1ScopePlan,
    _sha256_json,
    _stage1_scope_plan_body,
)
from .production_text_preparation import stable_file_sha256
from .review_spent_evidence_provider import (
    semantic_witness_config_from_portable_scientific_spec,
)
from .role_neutral_performance_benchmark import (
    RoleNeutralBenchmarkConfig,
    RoleNeutralBenchmarkSourceBinding,
    RoleNeutralBenchmarkWorkload,
)

ROLE_NEUTRAL_BENCHMARK_WORKLOAD_DEPLOYMENT_SCHEMA = (
    "portable_role_neutral_benchmark_workload_deployment_v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAUSED_PREFIX = (
    "input_preparation",
    "embedding_cache",
    "stage1_preflight",
)
_SEALED_PREPARED_CONTEXT_DIRECTORY = "sealed_prepared_stage1_context"
_SCOPE_KINDS = frozenset(
    {
        "full_outer",
        "exact_inner",
        "cumulative_spent",
    }
)


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"workload deployment contains duplicate key {key!r}")
        result[key] = value
    return result


def _absolute_path(value: Path | str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


@dataclass(frozen=True)
class RoleNeutralBenchmarkScopeSelector:
    """Configured content selector for one representative physical owner."""

    scope_label: str
    logical_scope_kind: str
    ordinal: int

    def __post_init__(self) -> None:
        label = str(self.scope_label).strip()
        if not label:
            raise ValueError("workload selector scope_label must be nonempty")
        if self.logical_scope_kind not in _SCOPE_KINDS:
            raise ValueError("workload selector has an unsupported logical_scope_kind")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("workload selector ordinal must be a nonnegative integer")
        object.__setattr__(self, "scope_label", label)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralBenchmarkScopeSelector":
        required = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "workload selector must configure every field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RoleNeutralBenchmarkWorkloadDeployment:
    """Path/config-only authority for reopening a paused workflow."""

    workflow_root: Path
    expected_workflow_request_sha256: str
    prepared_context_root: Path
    expected_benchmark_config_sha256: str
    representative_scope_selectors: tuple[
        RoleNeutralBenchmarkScopeSelector,
        ...,
    ]
    schema_version: str = ROLE_NEUTRAL_BENCHMARK_WORKLOAD_DEPLOYMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ROLE_NEUTRAL_BENCHMARK_WORKLOAD_DEPLOYMENT_SCHEMA:
            raise ValueError("unsupported benchmark workload deployment schema")
        workflow_root = _absolute_path(
            self.workflow_root,
            label="workflow_root",
        )
        prepared_root = _absolute_path(
            self.prepared_context_root,
            label="prepared_context_root",
        )
        for label, value in (
            (
                "expected_workflow_request_sha256",
                self.expected_workflow_request_sha256,
            ),
            (
                "expected_benchmark_config_sha256",
                self.expected_benchmark_config_sha256,
            ),
        ):
            if _SHA256.fullmatch(str(value)) is None:
                raise ValueError(f"{label} must be one lowercase SHA-256")
        selectors = tuple(self.representative_scope_selectors)
        if (
            not selectors
            or any(not isinstance(value, RoleNeutralBenchmarkScopeSelector) for value in selectors)
            or len({value.scope_label for value in selectors}) != len(selectors)
        ):
            raise ValueError("workload deployment requires unique typed representative selectors")
        object.__setattr__(self, "workflow_root", workflow_root)
        object.__setattr__(self, "prepared_context_root", prepared_root)
        object.__setattr__(self, "representative_scope_selectors", selectors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_root": str(self.workflow_root),
            "expected_workflow_request_sha256": (self.expected_workflow_request_sha256),
            "prepared_context_root": str(self.prepared_context_root),
            "expected_benchmark_config_sha256": (self.expected_benchmark_config_sha256),
            "representative_scope_selectors": [
                {
                    "scope_label": value.scope_label,
                    "logical_scope_kind": value.logical_scope_kind,
                    "ordinal": value.ordinal,
                }
                for value in self.representative_scope_selectors
            ],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "RoleNeutralBenchmarkWorkloadDeployment":
        required = {field.name for field in fields(cls)}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError(
                "workload deployment must configure every field exactly; "
                f"missing={sorted(required - set(value))}, "
                f"extra={sorted(set(value) - required)}"
            )
        selectors = value.get("representative_scope_selectors")
        if not isinstance(selectors, list):
            raise TypeError("representative_scope_selectors must be a list")
        return cls(
            workflow_root=Path(str(value["workflow_root"])),
            expected_workflow_request_sha256=str(value["expected_workflow_request_sha256"]),
            prepared_context_root=Path(str(value["prepared_context_root"])),
            expected_benchmark_config_sha256=str(value["expected_benchmark_config_sha256"]),
            representative_scope_selectors=tuple(
                RoleNeutralBenchmarkScopeSelector.from_mapping(row) for row in selectors
            ),
            schema_version=str(value["schema_version"]),
        )

    @classmethod
    def from_json(
        cls,
        path: Path | str,
    ) -> "RoleNeutralBenchmarkWorkloadDeployment":
        source = Path(path)
        try:
            value = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"workload deployment contains non-finite {constant}")
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("workload deployment is not closed UTF-8 JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("workload deployment must contain one JSON object")
        return cls.from_mapping(value)


@dataclass(frozen=True)
class AuthenticatedPausedStage1Preflight:
    """Freshly reopened immutable request and exact completed prefix."""

    root: Path
    request: Mapping[str, Any]
    phases: Mapping[str, Mapping[str, Any]]


def _sealed_prepared_context_manifest_path(
    deployment: RoleNeutralBenchmarkWorkloadDeployment,
    *,
    require_existing: bool,
) -> Path:
    """Resolve the one permitted artifact below the configured scratch root."""

    if not isinstance(require_existing, bool):
        raise TypeError("require_existing must be boolean")
    root = deployment.prepared_context_root
    manifest = (
        root
        / _SEALED_PREPARED_CONTEXT_DIRECTORY
        / PREPARED_STAGE1_CONTEXT_MANIFEST_NAME
    )
    if not require_existing:
        return manifest
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError(
            "existing prepared_context_root must be one canonical directory"
        )
    children = tuple(root.iterdir())
    if (
        len(children) != 1
        or children[0].name != _SEALED_PREPARED_CONTEXT_DIRECTORY
        or children[0].is_symlink()
        or not children[0].is_dir()
        or children[0].resolve(strict=True) != children[0]
    ):
        raise ValueError(
            "existing prepared_context_root is partial, substituted, or has "
            "unregistered entries"
        )
    return manifest


def _validate_prepared_context_bindings(
    *,
    artifact: PreparedStage1ContextArtifact,
    stage1_build_options: Stage1BundleBuildOptions,
    architecture_profiles: Mapping[str, Any],
    runtime_compatibility_class: str,
) -> None:
    """Bind a sealed context to the exact current authenticated request."""

    if not isinstance(artifact, PreparedStage1ContextArtifact):
        raise TypeError("prepared context did not reopen as its typed artifact")
    expected_options = serialize_stage1_build_options(stage1_build_options)
    locators = artifact.execution_locators
    if locators.get("stage1_build_options") != expected_options:
        raise ValueError(
            "sealed prepared context Stage 1 request/config locators differ "
            "from the authenticated benchmark workflow"
        )
    if locators.get("architecture_profiles") != copy.deepcopy(
        dict(architecture_profiles)
    ):
        raise ValueError(
            "sealed prepared context architecture profiles differ from the "
            "authenticated benchmark workflow"
        )
    if locators.get("runtime_compatibility_class") != str(
        runtime_compatibility_class
    ):
        raise ValueError(
            "sealed prepared context runtime compatibility class differs "
            "from the authenticated benchmark workflow"
        )
    exact_request = locators.get("exact_stage1_request")
    if (
        not isinstance(exact_request, Mapping)
        or exact_request.get("request_sha256")
        != _sha256_json(
            {
                key: value
                for key, value in exact_request.items()
                if key != "request_sha256"
            }
        )
    ):
        raise ValueError("sealed prepared context exact Stage 1 request changed")


def _authenticate_paused_stage1_preflight(
    deployment: RoleNeutralBenchmarkWorkloadDeployment,
    *,
    require_fresh_prepared_context: bool = True,
) -> AuthenticatedPausedStage1Preflight:
    root = deployment.workflow_root
    if root.is_symlink() or not root.is_dir():
        raise ValueError("workflow_root must be one real existing directory")
    resolved_root = root.resolve(strict=True)
    if resolved_root != root:
        raise ValueError("workflow_root must be canonical and symlink-free")
    prepared_root = deployment.prepared_context_root
    if not isinstance(require_fresh_prepared_context, bool):
        raise TypeError("require_fresh_prepared_context must be boolean")
    if prepared_root.is_symlink():
        raise ValueError("prepared_context_root must be symlink-free")
    if require_fresh_prepared_context:
        if prepared_root.exists():
            raise FileExistsError("prepared_context_root must be fresh")
        prepared_parent = prepared_root.parent.resolve(strict=True)
        if prepared_parent != prepared_root.parent or not prepared_parent.is_dir():
            raise ValueError("prepared_context_root parent must be canonical")
    elif prepared_root.exists():
        if (
            not prepared_root.is_dir()
            or prepared_root.resolve(strict=True) != prepared_root
        ):
            raise ValueError(
                "existing prepared_context_root must be one canonical directory"
            )
    else:
        prepared_parent = prepared_root.parent.resolve(strict=True)
        if prepared_parent != prepared_root.parent or not prepared_parent.is_dir():
            raise ValueError("prepared_context_root parent must be canonical")
    if prepared_root == root or root in prepared_root.parents:
        raise ValueError("prepared_context_root must not mutate the immutable workflow tree")

    request_path = root / "immutable_run_request.json"
    request = _read_json_object(
        request_path,
        label="benchmark source immutable workflow request",
    )
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    expected_request = deployment.expected_workflow_request_sha256
    phase_sequence = request.get("phase_sequence")
    if (
        request.get("schema_version") != WORKFLOW_SCHEMA
        or request.get("request_sha256") != expected_request
        or _sha(request_body) != expected_request
        or not isinstance(phase_sequence, list)
        or tuple(phase_sequence[: len(_PAUSED_PREFIX)]) != _PAUSED_PREFIX
        or len(phase_sequence) <= len(_PAUSED_PREFIX)
    ):
        raise ValueError("benchmark workflow request or requested phase prefix changed")
    _revalidate_request_bound_external_inputs(request)

    phases = {
        phase: _validate_phase_manifest_from_paths(
            work_root=root,
            phase=phase,
            request_sha256=expected_request,
        )
        for phase in _PAUSED_PREFIX
    }
    for phase in phase_sequence[len(_PAUSED_PREFIX) :]:
        later_phase_root = root / "phases" / str(phase)
        if later_phase_root.exists() or later_phase_root.is_symlink():
            raise ValueError("benchmark source workflow advanced beyond stage1_preflight")

    progress = _read_json_object(
        root / "workflow_progress.json",
        label="paused workflow progress",
    )
    required_progress = {
        "schema_version",
        "request_sha256",
        "status",
        "phase_sequence",
        "planned_phase_count",
        "completed_phases",
        "completed_phase_count",
        "current_phase",
        "remaining_phase_count",
        "stage1_gpu_ids",
        "stage1_execution_device_count",
        "stage1_execution_profile",
        "stage1_scope_workers_per_gpu",
        "stage1_preflight_workers",
        "tfidf_workers",
        "updated_at",
        "error",
    }
    if (
        set(progress) != required_progress
        or progress.get("schema_version") != WORKFLOW_PROGRESS_SCHEMA
        or progress.get("request_sha256") != expected_request
        or progress.get("status") != "paused"
        or progress.get("phase_sequence") != phase_sequence
        or progress.get("planned_phase_count") != len(phase_sequence)
        or progress.get("completed_phases") != list(_PAUSED_PREFIX)
        or progress.get("completed_phase_count") != len(_PAUSED_PREFIX)
        or progress.get("current_phase") is not None
        or progress.get("remaining_phase_count") != len(phase_sequence) - len(_PAUSED_PREFIX)
        or progress.get("stage1_gpu_ids")
        != request.get("resolved_stage1_gpu_ids")
        or progress.get("stage1_execution_device_count")
        != request.get("stage1_execution_device_count")
        or progress.get("stage1_execution_profile")
        != request.get("stage1_execution_profile")
        or progress.get("stage1_scope_workers_per_gpu")
        != request.get("stage1_scope_workers_per_gpu")
        or progress.get("stage1_preflight_workers")
        != request.get("stage1_preflight_workers")
        or progress.get("tfidf_workers") != request.get("tfidf_workers")
        or progress.get("error") is not None
        or not isinstance(progress.get("updated_at"), str)
        or not progress["updated_at"]
    ):
        raise ValueError(
            "workflow is not an authenticated operational pause after " "stage1_preflight"
        )

    source_snapshot = request.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, Mapping):
            raise ValueError("workflow source snapshot identity is invalid")
        loaded_source_root = Path(__file__).resolve().parents[2]
        requested_source_root = Path(str(source_snapshot.get("root", ""))).resolve(strict=True)
        if loaded_source_root != requested_source_root:
            raise RuntimeError(
                "benchmark provider must execute from the workflow's immutable " "source snapshot"
            )
    return AuthenticatedPausedStage1Preflight(
        root=root,
        request=copy.deepcopy(dict(request)),
        phases=copy.deepcopy(phases),
    )


def _registered_phase_path(
    phase: Mapping[str, Any],
    raw_path: Any,
    *,
    label: str,
    expected_name: str | None = None,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} locator is missing")
    path = Path(raw_path)
    registered = {
        Path(str(row["path"])).resolve(strict=True)
        for row in phase.get("artifacts", ())
        if isinstance(row, Mapping)
    }
    resolved = path.resolve(strict=True)
    state = os.lstat(path)
    if (
        not path.is_absolute()
        or path.is_symlink()
        or resolved != path
        or not stat.S_ISREG(state.st_mode)
        or int(state.st_nlink) != 1
        or resolved not in registered
        or (expected_name is not None and resolved.name != expected_name)
    ):
        raise ValueError(f"{label} is not a registered terminal artifact")
    return resolved


def _registered_phase_directory(
    phase: Mapping[str, Any],
    raw_path: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} locator is missing")
    path = Path(raw_path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{label} is not an absolute real directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory")
    registered = {
        Path(str(row["path"])).resolve(strict=True)
        for row in phase.get("artifacts", ())
        if isinstance(row, Mapping)
    }
    actual: set[Path] = set()
    for value in resolved.rglob("*"):
        state = os.lstat(value)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError(f"{label} contains a symlink")
        if stat.S_ISDIR(state.st_mode):
            continue
        if not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
            raise ValueError(f"{label} contains a non-private file")
        canonical = value.resolve(strict=True)
        if canonical != value:
            raise ValueError(f"{label} contains a noncanonical file")
        actual.add(canonical)
    expected = {
        value
        for value in registered
        if value != resolved and resolved in value.parents
    }
    if not actual or actual != expected:
        raise ValueError(f"{label} is not fully registered by its phase")
    return resolved


def _embedding_chunk_configuration(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    portable = request.get("portable_scientific_spec")
    if not isinstance(portable, Mapping):
        raise ValueError("real role-neutral benchmark requires a typed portable scientific request")
    text_windows = portable.get("text_windows")
    if not isinstance(text_windows, Mapping):
        raise ValueError("portable request lacks its text-window configuration")
    encoder = SentenceEmbeddingEncoderSpec.from_mapping(text_windows.get("embedding_encoder"))
    scientific_fields = {
        "embedding_chunk_size_words": "chunk_size_words",
        "embedding_chunk_overlap_words": "chunk_overlap_words",
        "embedding_max_chunks": "max_chunks",
        "embedding_chunk_selection": "chunk_selection",
        "embedding_max_seq_length": "max_seq_length",
        "embedding_normalize": "normalize_embeddings",
    }
    values: dict[str, Any] = {}
    for request_name, output_name in scientific_fields.items():
        expected = text_windows.get(request_name)
        if request.get(request_name) != expected:
            raise ValueError(
                f"workflow request changed {request_name} outside its "
                "portable scientific identity"
            )
        values[output_name] = expected
    if request.get("embedding_encoder") != dict(text_windows["embedding_encoder"]):
        raise ValueError("workflow request embedding encoder differs from its scientific identity")
    values.update(
        encoder.as_configuration(normalize_embeddings=bool(values.pop("normalize_embeddings")))
    )
    return values


def _resolved_cache_relocation(
    *,
    authenticated: AuthenticatedPausedStage1Preflight,
    cache: Path,
    prepared: Path,
    chunk_configuration: Mapping[str, Any],
) -> Any | None:
    request = authenticated.request
    imported = request.get("embedding_cache_import")
    if imported is None:
        return None
    from .production_embedding_cache_relocation import (
        ProductionEmbeddingCacheRelocationOptions,
    )

    cache_phase = authenticated.phases["embedding_cache"]
    cache_result = cache_phase["result"]
    identity = cache_result.get("cache_identity")
    if cache_result.get("mode") != "authenticated_relocation" or not isinstance(identity, Mapping):
        raise ValueError("workflow cache-import request lacks its authenticated relocation result")
    target = Path(str(identity.get("root", ""))).resolve(strict=True)
    if cache.parent != target or prepared.parent.parent != target:
        raise ValueError("relocated cache paths differ from their sealed root")

    preparation_phase = authenticated.phases["input_preparation"]
    preparation_result = preparation_phase.get("result")
    if not isinstance(preparation_result, Mapping):
        raise ValueError("input-preparation phase result is invalid")
    output = preparation_result.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("input-preparation result lacks its output")
    fresh_prepared = _registered_phase_path(
        preparation_phase,
        output.get("path"),
        label="fresh prepared cohort",
    )
    preparation_manifests = [
        Path(str(row["path"])).resolve(strict=True)
        for row in preparation_phase["artifacts"]
        if Path(str(row["path"])).name == "preparation_manifest.json"
    ]
    if len(preparation_manifests) != 1:
        raise ValueError("input preparation has no unique registered manifest")
    fresh_manifest = preparation_manifests[0]

    source_cache = Path(str(imported)).resolve(strict=True)
    explicit_prepared = request.get("embedding_cache_import_source_prepared_path")
    explicit_manifest = request.get("embedding_cache_import_source_preparation_manifest_path")
    if (explicit_prepared is None) != (explicit_manifest is None):
        raise ValueError("cache import has a partial source-preparation locator")
    if explicit_prepared is not None:
        source_prepared = Path(str(explicit_prepared)).resolve(strict=True)
        source_manifest = Path(str(explicit_manifest)).resolve(strict=True)
    else:
        metadata = _read_json_object(
            source_cache / "metadata.json",
            label="source embedding-cache metadata",
        )
        provenance = metadata.get("production_provenance")
        dataset = provenance.get("dataset") if isinstance(provenance, Mapping) else None
        source_path = dataset.get("path") if isinstance(dataset, Mapping) else None
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("source embedding cache does not identify its prepared cohort")
        source_prepared = Path(source_path).resolve(strict=True)
        source_manifest = (source_prepared.parent / "preparation_manifest.json").resolve(
            strict=True
        )
    return ProductionEmbeddingCacheRelocationOptions(
        source_cache_dir=source_cache,
        source_prepared_cohort_path=source_prepared,
        source_preparation_manifest_path=source_manifest,
        fresh_prepared_cohort_path=fresh_prepared,
        fresh_preparation_manifest_path=fresh_manifest,
        local_model_path=Path(str(request["embedding_local_model_path"])).resolve(strict=True),
        target_dir=target,
        unit_id_column=str(request["unit_id_column"]),
        text_column=str(request["text_column"]),
        treatment_column=str(request["treatment_column"]),
        outcome_column=str(request["outcome_column"]),
        sentence_model_name=str(request["embedding_model_name"]),
        chunk_configuration=copy.deepcopy(dict(chunk_configuration)),
    )


def _stage1_build_options(
    *,
    authenticated: AuthenticatedPausedStage1Preflight,
    deployment: RoleNeutralBenchmarkWorkloadDeployment,
) -> Stage1BundleBuildOptions:
    request = authenticated.request
    cache_phase = authenticated.phases["embedding_cache"]
    cache_result = cache_phase.get("result")
    if (
        not isinstance(cache_result, Mapping)
        or cache_result.get("schema_version") != EMBEDDING_CACHE_PHASE_SCHEMA
    ):
        raise ValueError("embedding-cache phase result has an unsupported schema")
    cache = _registered_phase_directory(
        cache_phase,
        cache_result.get("cache_path"),
        label="embedding cache",
    )
    prepared = _registered_phase_path(
        cache_phase,
        cache_result.get("prepared_cohort_path"),
        label="cache-bound prepared cohort",
    )

    preflight_phase = authenticated.phases["stage1_preflight"]
    preflight_result = preflight_phase.get("result")
    if (
        not isinstance(preflight_result, Mapping)
        or preflight_result.get("schema_version") != STAGE1_PREFLIGHT_PHASE_SCHEMA
    ):
        raise ValueError("stage1_preflight phase result has an unsupported schema")
    profile = _registered_phase_path(
        preflight_phase,
        preflight_result.get("effective_profile_path"),
        label="effective Stage 1 profile",
        expected_name="effective_stage1_profile.json",
    )
    preflight_manifest = _registered_phase_path(
        preflight_phase,
        preflight_result.get("cluster_preflight_manifest_path"),
        label="cluster preflight manifest",
        expected_name="cluster_preflight_manifest.json",
    )
    state_manifest = _registered_phase_path(
        preflight_phase,
        preflight_result.get("cluster_preflight_state_bundle_manifest_path"),
        label="cluster preflight state bundle",
        expected_name="cluster_state_bundle_manifest.json",
    )
    if preflight_result.get("cluster_preflight_states_are_canonical_no_refit") is not True:
        raise ValueError("clustered preflight state is not the canonical no-refit result")

    chunk_configuration = _embedding_chunk_configuration(request)
    relocation = _resolved_cache_relocation(
        authenticated=authenticated,
        cache=cache,
        prepared=prepared,
        chunk_configuration=chunk_configuration,
    )
    portable = request.get("portable_scientific_spec")
    if not isinstance(portable, Mapping):
        raise ValueError("workflow request lacks its portable scientific identity")
    resolved_gpu_ids = tuple(int(value) for value in request.get("resolved_stage1_gpu_ids", ()))
    if resolved_gpu_ids != tuple(int(value) for value in request.get("stage1_gpu_ids", ())):
        raise ValueError("workflow request changed its resolved Stage 1 devices")
    resolved_query_devices = tuple(
        str(value) for value in request.get("resolved_query_devices", ())
    )
    if resolved_query_devices != tuple(str(value) for value in request.get("query_devices", ())):
        raise ValueError("workflow request changed its resolved query devices")
    recovery_root = authenticated.root / "recovery"
    return Stage1BundleBuildOptions(
        dataset_path=prepared,
        config_path=profile,
        embedding_cache_dir=cache,
        output_dir=deployment.prepared_context_root,
        unit_id_column=str(request["unit_id_column"]),
        initial_training_partitions=int(request["initial_training_partitions"]),
        physical_fit_identity=Stage1PhysicalFitIdentity.from_mapping(
            request.get("stage1_physical_fit_identity") or {}
        ),
        embedding_local_model_path=None,
        embedding_cache_output_dir=None,
        seed=int(request["seed"]),
        device=str(request["stage1_device"]),
        gpu_ids=resolved_gpu_ids,
        num_workers=int(request["num_workers"]),
        tfidf_workers=int(request["tfidf_workers"]),
        tfidf_parallel_backend=str(request["tfidf_parallel_backend"]),
        query_devices=resolved_query_devices,
        query_nuisance_folds=int(request["interaction_inner_folds"]),
        query_config_path=Path(str(request["query_profile_path"])).resolve(strict=True),
        resume=False,
        dry_run=False,
        embedding_cache_relocation=relocation,
        embedding_cache_configuration=chunk_configuration,
        semantic_witness_scientific_config=(
            semantic_witness_config_from_portable_scientific_spec(portable)
        ),
        scope_workers_per_gpu=int(request["stage1_scope_workers_per_gpu"]),
        preflight_workers=int(request["stage1_preflight_workers"]),
        cluster_preflight_manifest_path=preflight_manifest,
        cluster_preflight_state_bundle_manifest_path=state_manifest,
        stage1_scope_descriptor_root=(recovery_root / "descriptor").resolve(),
        stage1_scope_attempt_root=(recovery_root / "stage1_scope_attempts").resolve(),
        stage1_scope_progress_path=(recovery_root / "stage1_scope_progress.json").resolve(),
        portable_cluster_preflight_v2=True,
    )


def _one_physical_owner_plan(
    *,
    source: Stage1ScopePlan,
    selector: RoleNeutralBenchmarkScopeSelector,
    fit_row_count: int,
) -> Stage1ScopePlan:
    """Select one physical equivalence group by purpose, size, and ordinal."""

    matching = tuple(
        (owner, members)
        for owner, members in source.physical_scope_groups
        if owner.fit_row_count == int(fit_row_count)
        and any(member.scope_kind == selector.logical_scope_kind for member in members)
    )
    if selector.ordinal >= len(matching):
        raise ValueError(
            f"selector {selector.scope_label!r} has no physical owner at "
            f"ordinal {selector.ordinal} for configured purpose and row count"
        )
    _owner, members = matching[selector.ordinal]
    member_ids = {value.scope_id for value in members}
    assignments = tuple(value for value in source.assignments if value.scope_id in member_ids)
    if not assignments:
        raise RuntimeError("selected physical group has no scheduler assignments")
    body = _stage1_scope_plan_body(
        registry_content_sha256=source.registry_content_sha256,
        global_seed=source.global_seed,
        review_rounds=source.review_rounds,
        initial_training_partitions=source.initial_training_partitions,
        physical_fit_identity=source.physical_fit_identity,
        gpu_ids=source.gpu_ids,
        scope_workers_per_gpu=source.scope_workers_per_gpu,
        scopes=members,
        assignments=assignments,
    )
    plan = Stage1ScopePlan(
        registry_content_sha256=source.registry_content_sha256,
        global_seed=source.global_seed,
        review_rounds=source.review_rounds,
        initial_training_partitions=source.initial_training_partitions,
        physical_fit_identity=source.physical_fit_identity,
        gpu_ids=source.gpu_ids,
        scope_workers_per_gpu=source.scope_workers_per_gpu,
        scopes=members,
        assignments=assignments,
        content_sha256=_sha256_json(body),
    )
    if (
        len(plan.physical_scopes) != 1
        or plan.physical_scopes[0].fit_row_count != int(fit_row_count)
        or not any(value.scope_kind == selector.logical_scope_kind for value in plan.scopes)
    ):
        raise RuntimeError("representative plan changed its selected content")
    plan.as_dict()
    return plan


def _immutable_inputs(
    authenticated: AuthenticatedPausedStage1Preflight,
) -> tuple[ImmutableInputObservation, ...]:
    identities: dict[str, int] = {}

    def register(sha256: Any, size: Any) -> None:
        digest = str(sha256)
        byte_count = int(size)
        if byte_count == 0:
            return
        if _SHA256.fullmatch(digest) is None or byte_count < 0:
            raise ValueError("immutable benchmark input identity is invalid")
        prior = identities.get(digest)
        if prior is not None and prior != byte_count:
            raise ValueError("immutable benchmark input hash has conflicting sizes")
        identities[digest] = byte_count

    for phase in _PAUSED_PREFIX:
        for row in authenticated.phases[phase]["artifacts"]:
            register(row["sha256"], row["size_bytes"])
    htr_tree = authenticated.request.get("htr_model_tree")
    htr_files = htr_tree.get("files") if isinstance(htr_tree, Mapping) else None
    if not isinstance(htr_files, list) or not htr_files:
        raise ValueError("workflow request lacks its authenticated HTR model files")
    for row in htr_files:
        if not isinstance(row, Mapping):
            raise ValueError("HTR model file inventory is invalid")
        register(row.get("sha256"), row.get("size_bytes"))
    if not identities:
        raise ValueError("benchmark workload has no immutable input bytes")
    return tuple(
        ImmutableInputObservation(
            content_sha256=digest,
            size_bytes=size,
        )
        for digest, size in sorted(identities.items())
    )


def build_authenticated_role_neutral_benchmark_workloads(
    config: RoleNeutralBenchmarkConfig,
    deployment_path: Path | str,
) -> Mapping[str, RoleNeutralBenchmarkWorkload]:
    """Return real typed workloads from one freshly authenticated pause."""

    if not isinstance(config, RoleNeutralBenchmarkConfig):
        raise TypeError("workload provider requires a typed benchmark config")
    deployment_source = Path(deployment_path)
    source_before = os.lstat(deployment_source)
    if (
        stat.S_ISLNK(source_before.st_mode)
        or not stat.S_ISREG(source_before.st_mode)
        or int(source_before.st_nlink) != 1
    ):
        raise ValueError("workload deployment must be one private regular file")
    deployment_resolved = deployment_source.resolve(strict=True)
    if (
        Path(os.path.abspath(os.fspath(deployment_source)))
        != deployment_resolved
    ):
        raise ValueError("workload deployment parent path must be symlink-free")
    deployment = RoleNeutralBenchmarkWorkloadDeployment.from_json(deployment_source)
    deployment_sha256, _deployment_size = stable_file_sha256(deployment_source)
    source_after = os.lstat(deployment_source)
    source_identity = lambda value: (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )
    if source_identity(source_before) != source_identity(source_after):
        raise RuntimeError("workload deployment changed while it was being parsed")
    if identity_sha256(config.as_dict()) != deployment.expected_benchmark_config_sha256:
        raise ValueError("benchmark config differs from workload deployment")
    configured_labels = {value.label for value in config.representative_scopes}
    selectors = {value.scope_label: value for value in deployment.representative_scope_selectors}
    if set(selectors) != configured_labels:
        raise ValueError("workload selectors do not match configured representative scopes")

    prepared_root_exists = (
        deployment.prepared_context_root.exists()
        or deployment.prepared_context_root.is_symlink()
    )
    authenticated = _authenticate_paused_stage1_preflight(
        deployment,
        require_fresh_prepared_context=not prepared_root_exists,
    )
    stage1_build_options = _stage1_build_options(
        authenticated=authenticated,
        deployment=deployment,
    )
    portable = authenticated.request.get("portable_scientific_spec")
    architecture_profiles = (
        portable.get("architecture_profiles") if isinstance(portable, Mapping) else None
    )
    if not isinstance(architecture_profiles, Mapping):
        raise ValueError("workflow request lacks all-ten architecture profiles")
    runtime_compatibility_class = str(
        authenticated.request["runtime_compatibility_class"]
    )
    factory_builder = PreparedBuildRoleNeutralProducerFactoriesBuilder(
        architecture_profiles=architecture_profiles,
        runtime_compatibility_class=runtime_compatibility_class,
    )
    if prepared_root_exists:
        prepared_context_manifest = (
            _sealed_prepared_context_manifest_path(
                deployment,
                require_existing=True,
            )
        )
        prepared_artifact = load_prepared_stage1_context(
            prepared_context_manifest
        )
        _validate_prepared_context_bindings(
            artifact=prepared_artifact,
            stage1_build_options=stage1_build_options,
            architecture_profiles=architecture_profiles,
            runtime_compatibility_class=runtime_compatibility_class,
        )
        prepared, _authenticated_factories = (
            prepared_artifact.reconstruct()
        )
        if (
            prepared.request
            != prepared_artifact.execution_locators[
                "exact_stage1_request"
            ]
            or serialize_stage1_build_options(prepared.options)
            != serialize_stage1_build_options(stage1_build_options)
        ):
            raise RuntimeError(
                "reconstructed prepared context differs from its exact "
                "authenticated request/config binding"
            )
    else:
        prepared = ProductionStage1BundleBuilder(
            stage1_build_options
        ).prepare()
        prepared_artifact = seal_prepared_stage1_context(
            root=(
                deployment.prepared_context_root
                / _SEALED_PREPARED_CONTEXT_DIRECTORY
            ),
            prepared=prepared,
            producer_factories_builder=factory_builder,
        )
        _sealed_prepared_context_manifest_path(
            deployment,
            require_existing=True,
        )
        _validate_prepared_context_bindings(
            artifact=prepared_artifact,
            stage1_build_options=stage1_build_options,
            architecture_profiles=architecture_profiles,
            runtime_compatibility_class=runtime_compatibility_class,
        )
    htr_profile = architecture_profiles.get("hierarchical_transformer")
    htr_producer_configuration = (
        htr_profile.get("producer_configuration")
        if isinstance(htr_profile, Mapping)
        else None
    )
    configured_htr_training_batch = (
        htr_producer_configuration.get("batch_size")
        if isinstance(htr_producer_configuration, Mapping)
        else None
    )
    prepared_htr_training_batch = prepared.config.training.batch_size
    if (
        isinstance(configured_htr_training_batch, bool)
        or not isinstance(configured_htr_training_batch, int)
        or configured_htr_training_batch < 1
        or isinstance(prepared_htr_training_batch, bool)
        or not isinstance(prepared_htr_training_batch, int)
        or prepared_htr_training_batch < 1
        or configured_htr_training_batch != prepared_htr_training_batch
    ):
        raise ValueError(
            "authenticated hierarchical-transformer optimizer batch differs "
            "from the prepared Stage 1 scientific profile"
        )
    immutable_inputs = _immutable_inputs(authenticated)
    maximum_concurrency_per_device = max(
        int(candidate.concurrency_per_device)
        for candidate in config.candidates
    )
    execution_profile = Stage1ExecutionProfile.from_mapping(
        authenticated.request["stage1_execution_profile"]
    )
    process_executor = (
        ProcessIsolatedRoleNeutralPhysicalOwnerExecutor(
            max_workers_per_resource=maximum_concurrency_per_device,
        ).bind_context(
            prepared_artifact.manifest_path,
        )
    )
    persistent_executor = (
        PersistentSpawnRoleNeutralPhysicalOwnerExecutor(
            max_workers_per_resource=maximum_concurrency_per_device,
            startup_timeout_seconds=(
                execution_profile.persistent_slot_startup_timeout_seconds
            ),
        ).bind_context(
            prepared_artifact.manifest_path,
        )
    )
    persistent_parameters = persistent_executor.worker_parameters
    if (
        not isinstance(persistent_parameters, Mapping)
        or set(persistent_parameters) != {"prepared_context_manifest_path"}
    ):
        raise RuntimeError(
            "persistent benchmark executor omitted its prepared-context binding"
        )
    rebound_manifest_path = Path(
        str(
            persistent_parameters[
                "prepared_context_manifest_path"
            ]
        )
    ).resolve(strict=True)
    if rebound_manifest_path != prepared_artifact.manifest_path:
        raise RuntimeError(
            "benchmark executor changed its sealed prepared-context binding"
        )
    scientific_identity = authenticated.request.get("scientific_identity")
    scientific_sha256 = (
        scientific_identity.get("scientific_sha256")
        if isinstance(scientific_identity, Mapping)
        else None
    )
    preflight_content_sha256 = authenticated.phases["stage1_preflight"].get(
        "content_sha256"
    )
    source_binding = RoleNeutralBenchmarkSourceBinding(
        workflow_request_sha256=deployment.expected_workflow_request_sha256,
        workflow_scientific_sha256=str(scientific_sha256),
        workload_deployment_sha256=deployment_sha256,
        stage1_preflight_phase_content_sha256=str(preflight_content_sha256),
        prepared_stage1_context_content_root_sha256=(
            prepared_artifact.content_root_sha256
        ),
    )

    def build_bound_executor(mode: str, workers: int) -> Any:
        if mode == "fresh_per_fit":
            selected = process_executor
        elif mode == "persistent_slots":
            selected = persistent_executor
        else:
            raise ValueError("benchmark executor mode is unsupported")
        return dataclasses.replace(
            selected,
            max_workers_per_resource=int(workers),
        )

    workloads: dict[str, RoleNeutralBenchmarkWorkload] = {}
    selected_owner_ids: set[str] = set()
    scope_by_label = {value.label: value for value in config.representative_scopes}
    for label in sorted(configured_labels):
        plan = _one_physical_owner_plan(
            source=prepared.stage1_scope_plan,
            selector=selectors[label],
            fit_row_count=scope_by_label[label].fit_row_count,
        )
        owner_id = plan.physical_scopes[0].scope_id
        if owner_id in selected_owner_ids:
            raise ValueError("representative selectors resolved to the same physical owner")
        selected_owner_ids.add(owner_id)

        def build_factories(
            *,
            binding: PreparedBuildRoleNeutralProducerFactoriesBuilder = (factory_builder),
            prepared_context: Any = prepared,
        ):
            return binding(prepared_context)

        workloads[label] = RoleNeutralBenchmarkWorkload(
            scope_label=label,
            plan=plan,
            scientific_htr_training_batch_size=(
                configured_htr_training_batch
            ),
            producer_factories_builder=build_factories,
            physical_owner_executor_builder=build_bound_executor,
            preflight_compression_source_builder=(
                lambda source=prepared.cluster_preflight_artifact_handle: source
            ),
            immutable_inputs=immutable_inputs,
            source_binding=source_binding,
        )
    return workloads


def write_authenticated_role_neutral_benchmark_workload_deployment(
    *,
    workflow_root: Path | str,
    benchmark_config: RoleNeutralBenchmarkConfig,
    prepared_context_root: Path | str,
    representative_scope_selectors: Sequence[RoleNeutralBenchmarkScopeSelector],
    output_path: Path | str,
) -> RoleNeutralBenchmarkWorkloadDeployment:
    """Authenticate a pause and publish its deterministic workload authority."""

    if not isinstance(benchmark_config, RoleNeutralBenchmarkConfig):
        raise TypeError("deployment writer requires a typed benchmark config")
    root = _absolute_path(workflow_root, label="workflow_root")
    request = _read_json_object(
        root / "immutable_run_request.json",
        label="workload deployment source request",
    )
    request_body = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    if (
        request.get("schema_version") != WORKFLOW_SCHEMA
        or _SHA256.fullmatch(str(request_sha256)) is None
        or _sha(request_body) != request_sha256
    ):
        raise ValueError("workload deployment source request is invalid")
    selectors = tuple(representative_scope_selectors)
    deployment = RoleNeutralBenchmarkWorkloadDeployment(
        workflow_root=root,
        expected_workflow_request_sha256=str(request_sha256),
        prepared_context_root=_absolute_path(
            prepared_context_root,
            label="prepared_context_root",
        ),
        expected_benchmark_config_sha256=identity_sha256(benchmark_config.as_dict()),
        representative_scope_selectors=selectors,
    )
    configured_labels = {value.label for value in benchmark_config.representative_scopes}
    if {
        value.scope_label for value in deployment.representative_scope_selectors
    } != configured_labels:
        raise ValueError("deployment writer selectors do not match benchmark scopes")
    _authenticate_paused_stage1_preflight(deployment)

    destination = _absolute_path(output_path, label="output_path")
    if destination == root or root in destination.parents:
        raise ValueError("workload deployment output must not mutate the immutable workflow tree")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("workload deployment output must be fresh")
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent or not parent.is_dir():
        raise ValueError("workload deployment output parent must be canonical")
    payload = (
        json.dumps(
            deployment.as_dict(),
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
                raise OSError("workload deployment write made no progress")
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
    reopened = RoleNeutralBenchmarkWorkloadDeployment.from_json(destination)
    if reopened != deployment:
        raise RuntimeError("published workload deployment changed on reopen")
    return reopened


__all__ = [
    "ROLE_NEUTRAL_BENCHMARK_WORKLOAD_DEPLOYMENT_SCHEMA",
    "AuthenticatedPausedStage1Preflight",
    "RoleNeutralBenchmarkScopeSelector",
    "RoleNeutralBenchmarkWorkloadDeployment",
    "build_authenticated_role_neutral_benchmark_workloads",
    "write_authenticated_role_neutral_benchmark_workload_deployment",
]
